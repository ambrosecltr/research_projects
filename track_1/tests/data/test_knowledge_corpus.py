from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from poetry50m.data.artifacts import read_prompt_records
from poetry50m.data.hf_sources import AcquiredArtifact, AcquiredSource, AcquisitionReceipt
from poetry50m.data.knowledge_corpus import build_knowledge_corpus
from poetry50m.data.loaders import iter_manifest


def _write_parquet(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(records), path)


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
    greats_path = acquisition / "poetry_greats" / "train.parquet"
    gutenberg_path = acquisition / "gutenberg_poetry" / "train.parquet"
    ultrafineweb_path = acquisition / "ultrafineweb_l3" / "part.parquet"
    _write_parquet(
        greats_path,
        [
            {
                "author": "Ada Poet",
                "book_title": "Collected Poems",
                "gutenberg_id": 11,
                "poem_title": "Rain Map",
                "poem_text": "Rain writes maps\non the window.",
                "line_count": 2,
                "word_count": 7,
            },
            {
                "author": "Ada Poet",
                "book_title": "Collected Poems",
                "gutenberg_id": 11,
                "poem_title": None,
                "poem_text": "An untitled line\nkeeps going.",
                "line_count": 2,
                "word_count": 5,
            },
            {
                "author": "Ada Poet",
                "book_title": "Collected Poems",
                "gutenberg_id": 11,
                "poem_title": None,
                "poem_text": "Editor's note: this is not a poem.",
                "line_count": 1,
                "word_count": 7,
            },
        ],
    )
    _write_parquet(
        gutenberg_path,
        [
            {"line": "A book line", "gutenberg_id": 11},
            {"line": "", "gutenberg_id": 11},
            {"line": "continues after a stanza break", "gutenberg_id": 11},
            {"line": "Another book", "gutenberg_id": 12},
            {"line": "*** END OF THE PROJECT GUTENBERG EBOOK", "gutenberg_id": 12},
        ],
    )
    _write_parquet(
        ultrafineweb_path,
        [
            {"uid": "record-1", "content": "A clear explanation of tides.", "style": "multi_style"},
            {"uid": "record-2", "content": "a clear explanation of tides.", "style": "multi_style"},
            {
                "uid": "record-3",
                "content": "Project Gutenberg boilerplate",
                "style": "multi_style",
            },
        ],
    )
    (acquisition / "acquisition_receipt.json").write_text("{}\n", encoding="utf-8")
    return AcquisitionReceipt(
        format_version=1,
        config_sha256="2" * 64,
        sources=(
            _acquired_source(
                source_id="poetry_greats",
                repository="yoonholee/poetry-greats-public-domain",
                artifact_kind="poetry_greats_parquet",
                artifact_path="data/train-00000-of-00001.parquet",
                local_path="poetry_greats/train.parquet",
                content=greats_path.read_bytes(),
            ),
            _acquired_source(
                source_id="gutenberg_poetry",
                repository="biglam/gutenberg-poetry-corpus",
                artifact_kind="gutenberg_poetry_parquet",
                artifact_path="data/train-00000-of-00001-fa9fb9e1f16eed7e.parquet",
                local_path="gutenberg_poetry/train.parquet",
                content=gutenberg_path.read_bytes(),
            ),
            _acquired_source(
                source_id="ultrafineweb_l3",
                repository="openbmb/Ultra-FineWeb-L3",
                artifact_kind="ultrafineweb_multistyle_parquet",
                artifact_path=(
                    "data/ultrafineweb_en_l3/multi_style/"
                    "part-00000-f36f5a53-4a77-434b-b9bc-67ed69b93fe2-c000.snappy.parquet"
                ),
                local_path="ultrafineweb_l3/part.parquet",
                content=ultrafineweb_path.read_bytes(),
            ),
        ),
    )


def _build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    acquisition = tmp_path / "acquired"
    receipt = _fixture_receipt(acquisition)
    monkeypatch.setattr(
        "poetry50m.data.knowledge_corpus.verify_acquisition",
        lambda _config, _directory: receipt,
    )
    sources_config = tmp_path / "sources.json"
    sources_config.write_text("{}\n", encoding="utf-8")
    selection_config = tmp_path / "selection.json"
    selection_config.write_text(
        json.dumps(
            {
                "format_version": 1,
                "ultrafineweb_l3": {
                    "max_documents": 200_000,
                    "method": "sha256_uid_priority_v1",
                    "seed": "fixture-selection-seed",
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "corpus"
    build_knowledge_corpus(
        acquisition_directory=acquisition,
        sources_config=sources_config,
        selection_config=selection_config,
        output_directory=output,
    )
    return output


def test_builds_three_source_corpus_with_explicit_training_roles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = _build(tmp_path, monkeypatch)

    documents = tuple(iter_manifest(output / "manifest.jsonl", allow_synthetic=False))
    assert [document.document_id for document in documents] == [
        "poetry-greats-00000000",
        "poetry-greats-00000001",
        "gutenberg-book-11",
        "gutenberg-book-12",
        "ultrafineweb-l3-record-1",
    ]
    greats = documents[:2]
    assert all(document.metadata["training_role"] == "conditional_poetry" for document in greats)
    assert [document.blocks[0].kind for document in greats] == ["poem", "poem"]
    assert documents[2].blocks[0].kind == "verse_document"
    assert documents[2].text == "A book line\n\ncontinues after a stanza break"
    assert documents[2].metadata["training_role"] == "unconditional_book_verse_ntp"
    assert documents[-1].metadata["training_role"] == "auxiliary_prose_ntp"
    assert documents[-1].provenance.rights_status == "unknown"

    prompts = read_prompt_records(output / "prompts.jsonl")
    assert [(prompt.method, prompt.prompt) for prompt in prompts] == [
        ("title", 'Write a poem titled "Rain Map".'),
        ("author_style", "Write a poem in the style of Ada Poet."),
    ]
    report = json.loads((output / "knowledge.report.json").read_text())
    assert report["rejection_counts"]["poetry_greats"] == {"editorial_residue": 1}
    assert report["rejection_counts"]["gutenberg_poetry"] == {"editorial_line": 1}
    assert report["rejection_counts"]["ultrafineweb_l3"] == {"editorial_residue": 1}
    assert report["exact_normalized_duplicates_removed"] == {"ultrafineweb_l3": 1}
    assert report["ultrafineweb_selection"]["eligible_count"] == 2
    assert report["ultrafineweb_selection"]["selected_count"] == 2
    assert report["leakage_families"] == {
        "gutenberg:11": [
            "poetry-greats-00000000",
            "poetry-greats-00000001",
            "gutenberg-book-11",
        ]
    }


def test_rejects_noncontiguous_gutenberg_book_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    acquisition = tmp_path / "acquired"
    receipt = _fixture_receipt(acquisition)
    path = acquisition / "gutenberg_poetry" / "train.parquet"
    _write_parquet(
        path,
        [
            {"line": "book eleven", "gutenberg_id": 11},
            {"line": "book twelve", "gutenberg_id": 12},
            {"line": "book eleven again", "gutenberg_id": 11},
        ],
    )
    monkeypatch.setattr(
        "poetry50m.data.knowledge_corpus.verify_acquisition",
        lambda _config, _directory: receipt,
    )
    sources_config = tmp_path / "sources.json"
    sources_config.write_text("{}\n", encoding="utf-8")
    selection_config = tmp_path / "selection.json"
    selection_config.write_text(
        '{"format_version":1,"ultrafineweb_l3":{"max_documents":1,"method":"sha256_uid_priority_v1","seed":"test"}}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must be contiguous"):
        build_knowledge_corpus(
            acquisition_directory=acquisition,
            sources_config=sources_config,
            selection_config=selection_config,
            output_directory=tmp_path / "corpus",
        )


def test_ultrafineweb_selection_is_hash_priority_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    acquisition = tmp_path / "acquired"
    receipt = _fixture_receipt(acquisition)
    monkeypatch.setattr(
        "poetry50m.data.knowledge_corpus.verify_acquisition",
        lambda _config, _directory: receipt,
    )
    sources_config = tmp_path / "sources.json"
    sources_config.write_text("{}\n", encoding="utf-8")
    seed = "selection-test"
    selection_config = tmp_path / "selection.json"
    selection_config.write_text(
        json.dumps(
            {
                "format_version": 1,
                "ultrafineweb_l3": {
                    "max_documents": 1,
                    "method": "sha256_uid_priority_v1",
                    "seed": seed,
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "corpus"

    build_knowledge_corpus(
        acquisition_directory=acquisition,
        sources_config=sources_config,
        selection_config=selection_config,
        output_directory=output,
    )

    expected_uid = min(
        ("record-1", "record-2"),
        key=lambda uid: hashlib.sha256(f"{seed}\0{uid}".encode()).digest(),
    )
    documents = tuple(iter_manifest(output / "manifest.jsonl", allow_synthetic=False))
    ultrafineweb_documents = [
        document for document in documents if document.document_id.startswith("ultrafineweb-l3-")
    ]
    assert [document.metadata["source_uid"] for document in ultrafineweb_documents] == [
        expected_uid
    ]
    report = json.loads((output / "knowledge.report.json").read_text())
    assert report["ultrafineweb_selection"]["max_documents"] == 1
    assert report["ultrafineweb_selection"]["eligible_count"] == 2
    assert report["ultrafineweb_selection"]["selected_count"] == 1


def test_rejects_schema_drift_in_poetry_greats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    acquisition = tmp_path / "acquired"
    receipt = _fixture_receipt(acquisition)
    path = acquisition / "poetry_greats" / "train.parquet"
    _write_parquet(path, [{"poem_text": "missing required fields"}])
    monkeypatch.setattr(
        "poetry50m.data.knowledge_corpus.verify_acquisition",
        lambda _config, _directory: receipt,
    )
    sources_config = tmp_path / "sources.json"
    sources_config.write_text("{}\n", encoding="utf-8")
    selection_config = tmp_path / "selection.json"
    selection_config.write_text(
        '{"format_version":1,"ultrafineweb_l3":{"max_documents":1,"method":"sha256_uid_priority_v1","seed":"test"}}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must contain exactly"):
        build_knowledge_corpus(
            acquisition_directory=acquisition,
            sources_config=sources_config,
            selection_config=selection_config,
            output_directory=tmp_path / "corpus",
        )
