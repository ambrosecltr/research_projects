from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from poetry50m.data.hf_sources import (
    AcquiredArtifact,
    AcquiredSource,
    AcquisitionReceipt,
)
from poetry50m.data.knowledge_corpus import build_knowledge_corpus
from poetry50m.data.loaders import iter_manifest


def _write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _acquired_source(
    *,
    source_id: str,
    repository: str,
    artifact_kind: str,
    artifact_path: str,
    local_path: str,
    content: bytes,
) -> AcquiredSource:
    revision = "1" * 40
    return AcquiredSource(
        source_id=source_id,
        repository=repository,
        revision=revision,
        resolved_revision=revision,
        artifact_kind=artifact_kind,
        artifacts=(
            AcquiredArtifact(
                source_path=artifact_path,
                local_path=local_path,
                sha256=hashlib.sha256(content).hexdigest(),
                size_bytes=len(content),
            ),
        ),
    )


def _fixture_receipt(acquisition: Path) -> AcquisitionReceipt:
    baby_path = acquisition / "babylm_distilled" / "babylm_cleaned.jsonl"
    nano_path = acquisition / "nano_wiki" / "nano_wiki_dataset.jsonl"
    _write_jsonl(
        baby_path,
        [
            {"text": "A clear explanation of how rain forms."},
            {"text": "A child asks why shadows move through the afternoon."},
        ],
    )
    _write_jsonl(
        nano_path,
        [{"title": "Tides", "text": "Tides rise and fall under lunar gravity."}],
    )
    (acquisition / "acquisition_receipt.json").write_text("{}\n", encoding="utf-8")
    return AcquisitionReceipt(
        format_version=1,
        config_sha256="2" * 64,
        sources=(
            _acquired_source(
                source_id="nano_wiki",
                repository="sixf0ur/nano_wiki",
                artifact_kind="synthetic_knowledge_jsonl",
                artifact_path="nano_wiki_dataset.jsonl",
                local_path="nano_wiki/nano_wiki_dataset.jsonl",
                content=nano_path.read_bytes(),
            ),
            _acquired_source(
                source_id="babylm_distilled",
                repository="sixf0ur/babylm_eng_distilled_1024",
                artifact_kind="distilled_text_jsonl",
                artifact_path="babylm_cleaned.jsonl",
                local_path="babylm_distilled/babylm_cleaned.jsonl",
                content=baby_path.read_bytes(),
            ),
        ),
    )


def test_builds_canonical_knowledge_corpus_with_explicit_rights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    acquisition = tmp_path / "acquired"
    receipt = _fixture_receipt(acquisition)
    monkeypatch.setattr(
        "poetry50m.data.knowledge_corpus.verify_acquisition",
        lambda _config, _directory: receipt,
    )
    sources_config = tmp_path / "sources.json"
    sources_config.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "corpus"

    receipt_path = build_knowledge_corpus(
        acquisition_directory=acquisition,
        sources_config=sources_config,
        output_directory=output,
    )

    documents = tuple(iter_manifest(output / "manifest.jsonl", allow_synthetic=False))
    assert [document.document_id for document in documents] == [
        "knowledge-babylm-distilled-00000000",
        "knowledge-babylm-distilled-00000001",
        "knowledge-nano-wiki-00000000",
    ]
    assert [document.provenance.rights_status for document in documents] == [
        "unknown",
        "unknown",
        "licensed",
    ]
    assert (output / "prompts.jsonl").read_text() == ""
    assert (output / "thoughts.jsonl").read_text() == ""
    assert (output / "pairings.jsonl").read_text() == ""
    assert json.loads(receipt_path.read_text())["format_version"] == 1


def test_rejects_drifted_source_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    acquisition = tmp_path / "acquired"
    receipt = _fixture_receipt(acquisition)
    baby_path = acquisition / "babylm_distilled" / "babylm_cleaned.jsonl"
    _write_jsonl(baby_path, [{"text": "Valid text.", "unexpected": "drift"}])
    monkeypatch.setattr(
        "poetry50m.data.knowledge_corpus.verify_acquisition",
        lambda _config, _directory: receipt,
    )
    sources_config = tmp_path / "sources.json"
    sources_config.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must contain exactly"):
        build_knowledge_corpus(
            acquisition_directory=acquisition,
            sources_config=sources_config,
            output_directory=tmp_path / "corpus",
        )
