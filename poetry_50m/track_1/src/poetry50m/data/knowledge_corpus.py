"""Build the pinned educational-prose corpus for the Track 1 8M lineage."""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import BinaryIO, cast

from poetry50m.config import canonical_json, file_hash
from poetry50m.trajectory._persistence import atomic_write

from .artifacts import write_pairings, write_prompt_records, write_thought_records
from .hf_sources import AcquiredSource, AcquisitionReceipt, verify_acquisition
from .schema import ContentBlock, Provenance, SourceDocument

_REMOVABLE_CONTROLS = (frozenset(range(32)) | frozenset(range(127, 160))).difference({9, 10})


@dataclass(frozen=True, slots=True)
class KnowledgeSource:
    source_id: str
    artifact_kind: str
    artifact_path: str


_SOURCES = (
    KnowledgeSource(
        source_id="babylm_distilled",
        artifact_kind="distilled_text_jsonl",
        artifact_path="babylm_cleaned.jsonl",
    ),
    KnowledgeSource(
        source_id="nano_wiki",
        artifact_kind="synthetic_knowledge_jsonl",
        artifact_path="nano_wiki_dataset.jsonl",
    ),
)


def _json_object(line: str, *, path: Path, line_number: int) -> dict[str, object]:
    try:
        value = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON at {path}:{line_number}") from error
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{path}:{line_number} must contain a JSON object")
    return cast(dict[str, object], value)


def _required_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _clean_text(text: str) -> str:
    normalized = unicodedata.normalize(
        "NFC",
        text.replace("\r\n", "\n").replace("\r", "\n"),
    )
    return "".join(
        character for character in normalized if ord(character) not in _REMOVABLE_CONTROLS
    ).strip()


def _records(path: Path) -> Iterator[tuple[int, dict[str, object]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL record at {path}:{line_number}")
            yield line_number, _json_object(line, path=path, line_number=line_number)


def _provenance(source: KnowledgeSource, acquired: AcquiredSource, *, title: str) -> Provenance:
    locator = f"{acquired.repository}@{acquired.revision}:{source.artifact_path}"
    if source.source_id == "nano_wiki":
        return Provenance(
            work=title,
            author="Gemma 3 27B; dataset curator David S.",
            licence="CC BY 4.0",
            source="sixf0ur/nano_wiki",
            source_locator=locator,
            rights_status="licensed",
            rights_evidence="https://huggingface.co/datasets/sixf0ur/nano_wiki",
            rights_notes="Synthetic educational article; attribution is required.",
        )
    return Provenance(
        work=title,
        author="DeepSeek; dataset curator sixf0ur",
        licence="Uploader declares CC0 1.0",
        source="sixf0ur/babylm_eng_distilled_1024",
        source_locator=locator,
        rights_status="unknown",
        rights_notes=(
            "The upload is marked CC0, but its BabyLM source has per-document licences "
            "and the distilled rows do not retain that source-level provenance."
        ),
    )


def _document(
    source: KnowledgeSource,
    acquired: AcquiredSource,
    *,
    row_index: int,
    record: Mapping[str, object],
) -> SourceDocument | None:
    expected_keys = {"title", "text"} if source.source_id == "nano_wiki" else {"text"}
    if set(record) != expected_keys:
        raise ValueError(
            f"{source.source_id} row {row_index} must contain exactly {sorted(expected_keys)}"
        )
    raw_text = _required_text(record["text"], name=f"{source.source_id} row text")
    text = _clean_text(raw_text)
    if not text or text == "[REJECT]":
        return None
    title = (
        _required_text(record["title"], name="nano_wiki title")
        if source.source_id == "nano_wiki"
        else f"Distilled BabyLM record {row_index}"
    )
    document_id = f"knowledge-{source.source_id.replace('_', '-')}-{row_index:08d}"
    block_id = f"{document_id}:paragraph:0"
    return SourceDocument(
        document_id=document_id,
        provenance=_provenance(source, acquired, title=title),
        text=text,
        blocks=(
            ContentBlock(
                block_id=block_id,
                kind="paragraph",
                text=text,
                paragraph_index=0,
                start_char=0,
                end_char=len(text),
            ),
        ),
        source_path=f"{source.source_id}/{source.artifact_path}",
        raw_text=raw_text,
        metadata={"source_row": str(row_index)},
        transformation_lineage=("utf8_decode", "unicode_nfc", "control_character_filter"),
    )


def _acquired_sources(receipt: AcquisitionReceipt) -> dict[str, AcquiredSource]:
    sources = {source.source_id: source for source in receipt.sources}
    expected = {source.source_id for source in _SOURCES}
    if set(sources) != expected:
        raise ValueError(
            f"acquisition source IDs differ: expected={sorted(expected)}, got={sorted(sources)}"
        )
    return sources


def _artifact_path(
    acquisition_directory: Path,
    source: KnowledgeSource,
    acquired: AcquiredSource,
) -> Path:
    if acquired.artifact_kind != source.artifact_kind or len(acquired.artifacts) != 1:
        raise ValueError(f"unexpected artifact contract for {source.source_id}")
    artifact = acquired.artifacts[0]
    if artifact.source_path != source.artifact_path:
        raise ValueError(f"unexpected artifact path for {source.source_id}")
    return acquisition_directory / artifact.local_path


def build_knowledge_corpus(
    *,
    acquisition_directory: Path,
    sources_config: Path,
    output_directory: Path,
) -> Path:
    """Write canonical prose documents plus a receipt from a verified acquisition."""
    receipt = verify_acquisition(sources_config, acquisition_directory)
    acquired_by_id = _acquired_sources(receipt)
    input_counts: dict[str, int] = {}
    document_counts: dict[str, int] = {}
    rejection_counts: dict[str, int] = {}
    duplicate_count = 0
    content_hashes: set[str] = set()

    manifest_path = output_directory / "manifest.jsonl"

    def write_manifest_stream(handle: BinaryIO) -> None:
        nonlocal duplicate_count
        for source in _SOURCES:
            acquired = acquired_by_id[source.source_id]
            path = _artifact_path(acquisition_directory, source, acquired)
            input_count = document_count = rejected_count = 0
            for row_index, record in _records(path):
                input_count += 1
                document = _document(
                    source,
                    acquired,
                    row_index=row_index - 1,
                    record=record,
                )
                if document is None:
                    rejected_count += 1
                    continue
                content_hash = sha256(document.text.casefold().encode()).hexdigest()
                if content_hash in content_hashes:
                    duplicate_count += 1
                    continue
                content_hashes.add(content_hash)
                document_count += 1
                handle.write(f"{canonical_json(document.to_mapping())}\n".encode())
            input_counts[source.source_id] = input_count
            document_counts[source.source_id] = document_count
            rejection_counts[source.source_id] = rejected_count

    atomic_write(manifest_path, write_manifest_stream)
    prompts_path = output_directory / "prompts.jsonl"
    thoughts_path = output_directory / "thoughts.jsonl"
    pairings_path = output_directory / "pairings.jsonl"
    write_prompt_records(prompts_path, ())
    write_thought_records(thoughts_path, ())
    write_pairings(pairings_path, ())

    report_path = output_directory / "knowledge.report.json"
    report = {
        "format_version": 1,
        "input_counts": input_counts,
        "document_counts": document_counts,
        "rejection_counts": rejection_counts,
        "exact_normalized_duplicates_removed": duplicate_count,
        "rights": {
            "nano_wiki": "CC BY 4.0 with attribution",
            "babylm_distilled": "unknown source-level rights; uploader declares CC0",
        },
    }
    report_path.write_text(f"{canonical_json(report)}\n", encoding="utf-8", newline="\n")
    receipt_path = output_directory / "knowledge.receipt.json"
    output_hashes = {
        path.name: file_hash(path)
        for path in (manifest_path, prompts_path, thoughts_path, pairings_path, report_path)
    }
    output_receipt = {
        "format_version": 1,
        "sources_config_sha256": file_hash(sources_config),
        "acquisition_receipt_sha256": file_hash(acquisition_directory / "acquisition_receipt.json"),
        "outputs": output_hashes,
    }
    receipt_path.write_text(
        f"{canonical_json(output_receipt)}\n",
        encoding="utf-8",
        newline="\n",
    )
    return receipt_path
