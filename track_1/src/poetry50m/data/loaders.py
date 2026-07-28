"""Deterministic loaders for text sources and JSONL corpus manifests."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from .schema import ContentBlock, Provenance, SourceDocument


def split_stanzas(text: str) -> tuple[str, ...]:
    """Return stanza payloads without claiming this structural view is the source."""
    normalised = text.replace("\r\n", "\n").replace("\r", "\n")
    return tuple(
        match.group(0)
        for match in re.finditer(r"[^\n](?:.*?[^\n])?(?=\n{2,}|\Z)", normalised, re.DOTALL)
    )


def paragraphs_to_document(
    *, document_id: str, text: str, provenance: Provenance, source_path: str = ""
) -> SourceDocument:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    spans = tuple(re.finditer(r"[^\n](?:.*?[^\n])?(?=\n{2,}|\Z)", cleaned, re.DOTALL))
    blocks = tuple(
        ContentBlock(
            block_id=f"{document_id}:paragraph:{index}",
            kind="paragraph",
            text=match.group(0),
            paragraph_index=index,
            start_char=match.start(),
            end_char=match.end(),
        )
        for index, match in enumerate(spans)
    )
    return SourceDocument(
        document_id=document_id,
        provenance=provenance,
        text=cleaned,
        blocks=blocks
        or (ContentBlock(block_id=f"{document_id}:document", kind="document", text=text),),
        source_path=source_path,
        raw_text=text,
        transformation_lineage=("utf8_decode", "newline_normalize"),
    )


def load_text_document(path: Path, *, document_id: str, provenance: Provenance) -> SourceDocument:
    """Load UTF-8 text or Markdown; Markdown remains source text, not rendered HTML."""
    if path.suffix.lower() not in {".txt", ".md", ".markdown"}:
        raise ValueError(f"expected .txt or Markdown source, got {path}")
    return paragraphs_to_document(
        document_id=document_id,
        text=path.read_text(encoding="utf-8"),
        provenance=provenance,
        source_path=str(path),
    )


def _document_from_record(record: dict[str, Any], *, source_path: str) -> SourceDocument:
    try:
        return SourceDocument.from_mapping(record)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid manifest record in {source_path}: {error}") from error


def iter_manifest(path: Path, *, allow_synthetic: bool = False) -> Iterator[SourceDocument]:
    """Read one fully structured source-document object per UTF-8 JSONL line."""
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                raise ValueError(f"blank manifest record at {path}:{line_number}")
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on {path}:{line_number}") from error
            if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
                raise ValueError(f"manifest record {path}:{line_number} must be an object")
            document = _document_from_record(value, source_path=f"{path}:{line_number}")
            if document.document_id in seen_ids:
                raise ValueError(f"duplicate document ID {document.document_id!r} in {path}")
            assert_ingestible(document, allow_synthetic=allow_synthetic)
            seen_ids.add(document.document_id)
            yield document


def assert_ingestible(document: SourceDocument, *, allow_synthetic: bool = False) -> None:
    """Preserve uncertain provenance; only explicit denial blocks personal research use."""
    status = document.provenance.rights_status
    if status == "denied":
        raise PermissionError(
            f"{document.document_id} cannot enter a corpus with rights_status={status}"
        )
    if status == "synthetic" and not allow_synthetic:
        raise PermissionError("synthetic content requires allow_synthetic=True")


def write_manifest(
    path: Path, documents: Iterable[SourceDocument], *, allow_synthetic: bool = False
) -> None:
    """Write canonical JSONL without discarding provenance or structural units."""
    document_list = sorted(documents, key=lambda document: document.document_id)
    if len({document.document_id for document in document_list}) != len(document_list):
        raise ValueError("cannot write a manifest with duplicate document IDs")
    for document in document_list:
        assert_ingestible(document, allow_synthetic=allow_synthetic)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for document in document_list:
            handle.write(json.dumps(document.to_mapping(), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
