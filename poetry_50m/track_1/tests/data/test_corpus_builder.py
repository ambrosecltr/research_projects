from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pyarrow as arrow  # type: ignore[import-untyped]
import pyarrow.parquet as parquet  # type: ignore[import-untyped]
import pytest

import poetry50m.data.corpus_builder as corpus_builder_module
from poetry50m.cli import main
from poetry50m.config import canonical_json, file_hash
from poetry50m.data.artifacts import (
    read_pairings,
    read_prompt_records,
    read_thought_records,
)
from poetry50m.data.corpus_builder import (
    build_corpus,
    load_standard_ebooks_selection,
)
from poetry50m.data.hf_sources import load_hf_sources_config
from poetry50m.data.loaders import iter_manifest

_CONFIG_SHA256 = "e555f0b1054e91fffe08d85e8a764c663959af0a71107bbb74884aef461dcb20"
_REVISIONS = {
    "public_domain_poetry": "84a87909d09ff0c3ae040c4e0af25a6344d96531",
    "poetry_greats_public_domain": "3201e250462905a0c8f6134e124382ac96586dc9",
    "standardebooks": "a2bafeeff73d3ff553e29dffc54f07772472b409",
}


def test_builder_contract_matches_frozen_acquisition_config() -> None:
    project_root = Path(__file__).parents[2]
    config = load_hf_sources_config(project_root / "configs/data/huggingface_sources.json")
    assert config.sha256 == corpus_builder_module._APPROVED_ACQUISITION_CONFIG_SHA256
    assert corpus_builder_module._EXPECTED_SOURCES == {
        source.source_id: (
            source.repository,
            source.revision,
            source.artifact_kind,
            {
                f"{source.source_id}/{artifact.path}": (
                    artifact.path,
                    artifact.sha256,
                    artifact.size_bytes,
                )
                for artifact in source.artifacts
            },
        )
        for source in config.sources
    }


def _write_parquet(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    parquet.write_table(arrow.Table.from_pylist(rows), path)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


def _artifact(root: Path, source_path: str, local_path: str) -> dict[str, object]:
    path = root / local_path
    return {
        "source_path": source_path,
        "local_path": local_path,
        "sha256": file_hash(path),
        "size_bytes": path.stat().st_size,
    }


def _source(
    root: Path,
    *,
    source_id: str,
    repository: str,
    artifact_kind: str,
    paths: tuple[tuple[str, str], ...],
) -> dict[str, object]:
    revision = _REVISIONS[source_id]
    return {
        "source_id": source_id,
        "repository": repository,
        "revision": revision,
        "resolved_revision": revision,
        "artifact_kind": artifact_kind,
        "artifacts": [
            _artifact(root, source_path, local_path) for source_path, local_path in paths
        ],
    }


def _yoon_poem(
    *,
    author: str,
    title: str | None,
    text: str,
    gutenberg_id: int,
) -> dict[str, object]:
    return {
        "author": author,
        "book_title": "Test Edition",
        "gutenberg_id": gutenberg_id,
        "poem_title": title,
        "poem_text": text,
        "line_count": len(text.splitlines()),
        "word_count": len(text.split()),
    }


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    selection: list[dict[str, str]] | None = None,
    duplicate_standard_work: bool = False,
    extra_dan_rows: list[dict[str, object]] | None = None,
    extra_yoon_rows: list[dict[str, object]] | None = None,
    first_standard_text: str | None = None,
    second_standard_text: str | None = None,
) -> tuple[Path, Path, Path]:
    acquisition = tmp_path / "acquired"
    shared_dan = "Moon over water  \r\n\r\nNight remembers.\r\n"
    dan_rows: list[dict[str, object]] = [
        {"Author": "Dan Author", "Title": "Dan Title", "text": shared_dan},
        {
            "Author": "Anonymous",
            "Title": "",
            "text": "A nameless bell sounds\nthrough the winter field.",
        },
        {"Author": "Nobody", "Title": "Empty", "text": " \n\n"},
    ]
    dan_rows.extend(extra_dan_rows or ())
    _write_json(
        acquisition / "public_domain_poetry" / "poems.json",
        dan_rows,
    )
    yoon_rows: list[dict[str, object]] = [
        {
            "author": "Yoon Author",
            "book_title": "Gold Edition",
            "gutenberg_id": 123,
            "poem_title": "Yoon Preferred",
            "poem_text": "Moon over water\n\nNight remembers.",
            "line_count": 2,
            "word_count": 5,
        },
        {
            "author": "Editor",
            "book_title": "Gold Edition",
            "gutenberg_id": 123,
            "poem_title": None,
            "poem_text": (
                "From the handwriting of the author, this transcription note "
                "describes a damaged page and is not a poem."
            ),
            "line_count": 1,
            "word_count": 18,
        },
    ]
    yoon_rows.extend(extra_yoon_rows or ())
    _write_parquet(
        acquisition / "poetry_greats_public_domain" / "data" / "train-00000-of-00001.parquet",
        yoon_rows,
    )
    first_standard_rows: list[dict[str, object]] = [
        {
            "link": "https://standardebooks.org/ebooks/author/first-work",
            "title": "First Work",
            "author": "Author One",
            "text": first_standard_text
            or (
                "Attention waits beside the river until the ordinary world grows clear.\n\n"
                "A second paragraph keeps the selected work above the minimum length."
            ),
            "language": "en-US",
        },
        {
            "link": "https://standardebooks.org/ebooks/author/unselected",
            "title": "Unselected Work",
            "author": "Other Author",
            "text": "This unselected book remains outside the built corpus. " * 4,
            "language": "en-US",
        },
    ]
    second_standard_rows: list[dict[str, object]] = [
        {
            "link": "https://standardebooks.org/ebooks/author/second-work",
            "title": "Second Work",
            "author": "Author Two",
            "text": second_standard_text
            or (
                "The mind becomes quiet by returning to what is immediately present.\n\n"
                "Another paragraph supplies enough contemplative prose for the fixture."
            ),
            "language": "en-GB",
        }
    ]
    if duplicate_standard_work:
        second_standard_rows.append(dict(first_standard_rows[0]))
    _write_parquet(
        acquisition / "standardebooks" / "data" / "train-00000-of-00002.parquet",
        first_standard_rows,
    )
    _write_parquet(
        acquisition / "standardebooks" / "data" / "train-00001-of-00002.parquet",
        second_standard_rows,
    )
    sources = [
        _source(
            acquisition,
            source_id="poetry_greats_public_domain",
            repository="yoonholee/poetry-greats-public-domain",
            artifact_kind="poem_records_parquet",
            paths=(
                (
                    "data/train-00000-of-00001.parquet",
                    "poetry_greats_public_domain/data/train-00000-of-00001.parquet",
                ),
            ),
        ),
        _source(
            acquisition,
            source_id="public_domain_poetry",
            repository="DanFosing/public-domain-poetry",
            artifact_kind="poem_records_json",
            paths=(("poems.json", "public_domain_poetry/poems.json"),),
        ),
        _source(
            acquisition,
            source_id="standardebooks",
            repository="Nelathan/standardebooks",
            artifact_kind="ebook_records_parquet",
            paths=(
                (
                    "data/train-00000-of-00002.parquet",
                    "standardebooks/data/train-00000-of-00002.parquet",
                ),
                (
                    "data/train-00001-of-00002.parquet",
                    "standardebooks/data/train-00001-of-00002.parquet",
                ),
            ),
        ),
    ]
    monkeypatch.setattr(
        corpus_builder_module,
        "_EXPECTED_SOURCES",
        {
            source["source_id"]: (
                source["repository"],
                source["revision"],
                source["artifact_kind"],
                {
                    artifact["local_path"]: (
                        artifact["source_path"],
                        artifact["sha256"],
                        artifact["size_bytes"],
                    )
                    for artifact in cast(list[dict[str, object]], source["artifacts"])
                },
            )
            for source in sources
        },
    )
    catalog = acquisition / "acquisition_receipt.json"
    catalog.write_text(
        canonical_json(
            {
                "format_version": 1,
                "config_sha256": _CONFIG_SHA256,
                "sources": sources,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    selection_path = tmp_path / "selection.json"
    _write_json(
        selection_path,
        {
            "format_version": 1,
            "standard_ebooks": {
                "works": selection
                if selection is not None
                else [
                    {"title": "First Work", "author": "Author One"},
                    {"title": "Second Work", "author": "Author Two"},
                ]
            },
        },
    )
    return acquisition, catalog, selection_path


def test_build_corpus_deduplicates_with_yoon_provenance_and_reports_rejections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    acquisition, catalog, selection = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "corpus"
    artifact = build_corpus(
        acquisition_directory=acquisition,
        catalog_path=catalog,
        selection_config_path=selection,
        output_directory=output,
    )

    documents = tuple(iter_manifest(output / "manifest.jsonl"))
    poems = tuple(document for document in documents if document.document_id.startswith("poem-"))
    prose = tuple(document for document in documents if document.document_id.startswith("prose-"))
    assert len(poems) == 2
    assert len(prose) == 2
    shared = next(document for document in poems if document.provenance.work == "Yoon Preferred")
    assert shared.provenance.author == "Yoon Author"
    assert shared.provenance.source == "yoonholee/poetry-greats-public-domain"
    assert shared.provenance.rights_status == "unknown"
    assert shared.provenance.rights_evidence.endswith(
        f"/tree/{_REVISIONS['poetry_greats_public_domain']}"
    )
    assert "United States" in shared.provenance.rights_notes
    assert "every jurisdiction" in shared.provenance.rights_notes
    assert shared.metadata["origin_count"] == "2"
    origins = json.loads(shared.metadata["origins_json"])
    assert [origin["source_id"] for origin in origins] == [
        "poetry_greats_public_domain",
        "public_domain_poetry",
    ]
    assert shared.text == "Moon over water\n\nNight remembers."
    assert shared.raw_text == "Moon over water\n\nNight remembers."
    assert shared.metadata["raw_content_sha256"] == shared.raw_content_hash
    assert shared.metadata["cleaned_content_sha256"] == shared.cleaned_content_hash
    assert {block.kind for block in shared.blocks} == {"poem", "stanza"}

    prompts = read_prompt_records(output / "prompts.jsonl")
    assert len(prompts) == 2
    assert any(prompt.prompt == "Write a poem titled: Yoon Preferred" for prompt in prompts)
    generic = next(prompt for prompt in prompts if prompt.prompt == "Write a poem.")
    assert "neither its source title nor author" in generic.source_attribution
    assert read_thought_records(output / "thoughts.jsonl") == ()
    assert read_pairings(output / "pairings.jsonl") == ()
    assert all(block.kind == "paragraph" for document in prose for block in document.blocks)
    assert all(document.provenance.rights_status == "unknown" for document in prose)

    report = artifact.report
    assert report["exact_duplicate_poetry_rows_removed"] == 1
    assert report["rejection_counts"] == {"editorial_note": 1, "empty_text": 1}
    assert report["document_counts"] == {"poetry": 2, "prose": 2, "total": 4}
    assert report["prompt_strategy_counts"] == {"generic": 1, "title": 1}
    assert artifact.receipt["counts"] == {
        "documents": 4,
        "prompts": 2,
        "thoughts": 0,
        "pairings": 0,
    }
    assert (output / "thoughts.jsonl").read_bytes() == b""
    assert (output / "pairings.jsonl").read_bytes() == b""

    second_output = tmp_path / "corpus-again"
    build_corpus(
        acquisition_directory=acquisition,
        catalog_path=catalog,
        selection_config_path=selection,
        output_directory=second_output,
    )
    for name in (
        "manifest.jsonl",
        "prompts.jsonl",
        "thoughts.jsonl",
        "pairings.jsonl",
        "corpus.report.json",
        "corpus.receipt.json",
    ):
        assert (output / name).read_bytes() == (second_output / name).read_bytes()


def test_poetry_filters_only_non_poems_and_preserves_long_or_sanitizable_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    long_text = ("long " * 30_001).strip()
    control_text = "Stars\x00 gather\x1f beneath\x7f a blue\x85 moon."
    acquisition, catalog, selection = _fixture(
        tmp_path,
        monkeypatch,
        extra_dan_rows=[
            {"Author": "Control Poet", "Title": "Signal", "text": control_text},
            {"Author": "Long Poet", "Title": "Long Work", "text": long_text},
            {
                "Author": "Prose Writer",
                "Title": "A Christmas Idyll (Prose)",
                "text": "This is a prose story presented as a poetry record.",
            },
            {
                "Author": "Fable Writer",
                "Title": "Education (Prose Fable)",
                "text": "This prose fable is explicitly labelled by its source title.",
            },
        ],
        extra_yoon_rows=[
            _yoon_poem(
                author="Editor",
                title=None,
                text="NOTES:\n_63. The source editor explains this textual variant._",
                gutenberg_id=124,
            )
        ],
    )
    output = tmp_path / "corpus"
    artifact = build_corpus(
        acquisition_directory=acquisition,
        catalog_path=catalog,
        selection_config_path=selection,
        output_directory=output,
    )

    poems = tuple(
        document
        for document in iter_manifest(output / "manifest.jsonl")
        if document.document_id.startswith("poem-")
    )
    assert len(poems) == 4
    signal = next(document for document in poems if document.provenance.work == "Signal")
    assert signal.raw_text == control_text
    assert signal.text == "Stars gather beneath a blue moon."
    assert signal.metadata["removed_control_character_count"] == "4"
    assert "strip_removable_control_characters" in signal.transformation_lineage
    long_poem = next(document for document in poems if document.provenance.work == "Long Work")
    assert long_poem.text == long_text
    assert len(long_poem.text) > 100_000
    assert "\n" not in long_poem.text

    assert artifact.report["rejection_counts"] == {
        "editorial_note": 1,
        "editorial_notes": 1,
        "empty_text": 1,
        "title_labelled_prose": 2,
    }
    filters = artifact.report["filters"]
    assert isinstance(filters, dict)
    assert filters["common_rejections"] == [
        "empty_text",
        "no_alphabetic_content",
    ]
    assert filters["poetry_rejections"] == [
        "editorial_note",
        "editorial_notes",
        "title_labelled_prose",
    ]
    assert artifact.report["transformation_counts"] == {
        "control_character_rows_sanitized": 1,
        "control_character_rows_sanitized_by_source": {
            "poetry_greats_public_domain": 0,
            "public_domain_poetry": 1,
            "standardebooks": 0,
        },
        "standard_ebooks_leading_introductions_removed": 0,
    }
    serialized_filters = json.dumps(filters)
    assert "too_long" not in serialized_filters
    assert "extreme_unit_length" not in serialized_filters
    assert "control_character" not in serialized_filters


def test_prompts_use_clean_titles_then_author_fallback_then_generic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    overlong_title = "L" * 121
    acquisition, catalog, selection = _fixture(
        tmp_path,
        monkeypatch,
        extra_dan_rows=[
            {"Author": "Roman Poet", "Title": "VI", "text": "Roman bells cross the field."},
            {"Author": "Number Poet", "Title": "39", "text": "Thirty nine birds rise."},
            {
                "Author": "Bracket Poet",
                "Title": "[Editorial marker]",
                "text": "A bracketed source marker falls away.",
            },
            {
                "Author": "Preface Poet",
                "Title": "Preface",
                "text": "The actual poem follows an unsuitable title.",
            },
            {
                "Author": "Long Title Poet",
                "Title": overlong_title,
                "text": "A short poem carried an impractically long title.",
            },
            {
                "Author": "Named Poet",
                "Title": "",
                "text": "An absent title still has a meaningful author.",
            },
            {
                "Author": "Anonymous",
                "Title": "",
                "text": "Neither title nor named author can condition this poem.",
            },
            {
                "Author": "Quote Poet",
                "Title": 'A "Quoted" Sky',
                "text": "Quotation marks belong inside this meaningful title.",
            },
        ],
    )
    output = tmp_path / "corpus"
    artifact = build_corpus(
        acquisition_directory=acquisition,
        catalog_path=catalog,
        selection_config_path=selection,
        output_directory=output,
    )

    documents = {
        document.document_id: document
        for document in iter_manifest(output / "manifest.jsonl")
        if document.document_id.startswith("poem-")
    }
    prompts = read_prompt_records(output / "prompts.jsonl")
    prompts_by_author = {
        documents[prompt.document_id].provenance.author: prompt.prompt for prompt in prompts
    }
    methods_by_author = {
        documents[prompt.document_id].provenance.author: prompt.method for prompt in prompts
    }
    assert prompts_by_author["Yoon Author"] == "Write a poem titled: Yoon Preferred"
    assert prompts_by_author["Quote Poet"] == 'Write a poem titled: A "Quoted" Sky'
    assert methods_by_author["Yoon Author"] == "title"
    assert methods_by_author["Quote Poet"] == "title"
    for author in (
        "Roman Poet",
        "Number Poet",
        "Bracket Poet",
        "Preface Poet",
        "Long Title Poet",
        "Named Poet",
    ):
        assert prompts_by_author[author] == f"Write a poem in the style of {author}."
        assert methods_by_author[author] == "author_style"
    assert sum(prompt.prompt == "Write a poem." for prompt in prompts) == 2
    assert {prompt.method for prompt in prompts if prompt.prompt == "Write a poem."} == {"generic"}
    assert artifact.report["prompt_strategy_counts"] == {
        "author_style": 6,
        "generic": 2,
        "title": 2,
    }


def test_standard_ebooks_leading_introduction_is_removed_with_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_first = (
        "## Introduction\n"
        "A translator\x0e explains the edition.\n\n"
        "### A note within the introduction\n"
        "This lower-level heading remains part of the introduction.\n\n"
        "## Book I\n"
        "Attention opens at the gate.\n\n"
        "The first discourse begins here."
    )
    second_text = (
        "## I\nThe song begins without an introduction.\n\nIts second paragraph remains unchanged."
    )
    acquisition, catalog, selection = _fixture(
        tmp_path,
        monkeypatch,
        first_standard_text=raw_first,
        second_standard_text=second_text,
    )
    output = tmp_path / "corpus"
    artifact = build_corpus(
        acquisition_directory=acquisition,
        catalog_path=catalog,
        selection_config_path=selection,
        output_directory=output,
    )

    prose = {
        document.provenance.work: document
        for document in iter_manifest(output / "manifest.jsonl")
        if document.document_id.startswith("prose-")
    }
    first = prose["First Work"]
    assert first.raw_text == raw_first
    assert first.text == (
        "## Book I\nAttention opens at the gate.\n\nThe first discourse begins here."
    )
    assert first.metadata["removed_control_character_count"] == "1"
    assert first.metadata["leading_markdown_introduction_removed"] == "true"
    assert "strip_removable_control_characters" in first.transformation_lineage
    assert "strip_leading_markdown_introduction" in first.transformation_lineage
    second = prose["Second Work"]
    assert second.text == second_text
    assert second.metadata["leading_markdown_introduction_removed"] == "false"
    assert "strip_leading_markdown_introduction" not in second.transformation_lineage
    assert artifact.report["transformation_counts"] == {
        "control_character_rows_sanitized": 1,
        "control_character_rows_sanitized_by_source": {
            "poetry_greats_public_domain": 0,
            "public_domain_poetry": 0,
            "standardebooks": 1,
        },
        "standard_ebooks_leading_introductions_removed": 1,
    }
    unbounded_introduction = "## Introduction\nThere is no following level-two heading."
    assert corpus_builder_module._strip_leading_markdown_introduction(unbounded_introduction) == (
        unbounded_introduction,
        False,
    )


def test_build_corpus_binds_approved_revision_and_artifact_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    acquisition, catalog, selection = _fixture(tmp_path, monkeypatch)
    value = json.loads(catalog.read_text(encoding="utf-8"))
    value["sources"][0]["revision"] = "0" * 40
    value["sources"][0]["resolved_revision"] = "0" * 40
    tampered = acquisition / "tampered_receipt.json"
    tampered.write_text(canonical_json(value) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="approved resolved revision"):
        build_corpus(
            acquisition_directory=acquisition,
            catalog_path=tampered,
            selection_config_path=selection,
            output_directory=tmp_path / "tampered-output",
        )

    original = acquisition / "public_domain_poetry" / "poems.json"
    original.write_text(original.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match its receipt"):
        build_corpus(
            acquisition_directory=acquisition,
            catalog_path=catalog,
            selection_config_path=selection,
            output_directory=tmp_path / "hash-output",
        )

    rebound = json.loads(catalog.read_text(encoding="utf-8"))
    dan_source = next(
        source for source in rebound["sources"] if source["source_id"] == "public_domain_poetry"
    )
    dan_source["artifacts"][0]["sha256"] = file_hash(original)
    dan_source["artifacts"][0]["size_bytes"] = original.stat().st_size
    rebound_catalog = acquisition / "rebound_receipt.json"
    rebound_catalog.write_text(canonical_json(rebound) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact identity is not approved"):
        build_corpus(
            acquisition_directory=acquisition,
            catalog_path=rebound_catalog,
            selection_config_path=selection,
            output_directory=tmp_path / "rebound-output",
        )


def test_standard_ebooks_selection_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    duplicate_config = tmp_path / "duplicate.json"
    duplicate = {"title": "First Work", "author": "Author One"}
    _write_json(
        duplicate_config,
        {
            "format_version": 1,
            "standard_ebooks": {"works": [duplicate, duplicate]},
        },
    )
    with pytest.raises(ValueError, match="duplicate title-and-author"):
        load_standard_ebooks_selection(duplicate_config)

    unknown_key_config = tmp_path / "unknown.json"
    _write_json(
        unknown_key_config,
        {
            "format_version": 1,
            "standard_ebooks": {"works": [duplicate], "titles": ["First Work"]},
        },
    )
    with pytest.raises(ValueError, match="must contain exactly"):
        load_standard_ebooks_selection(unknown_key_config)

    acquisition, catalog, missing_selection = _fixture(
        tmp_path / "missing",
        monkeypatch,
        selection=[{"title": "Absent Work", "author": "Absent Author"}],
    )
    with pytest.raises(ValueError, match="missing"):
        build_corpus(
            acquisition_directory=acquisition,
            catalog_path=catalog,
            selection_config_path=missing_selection,
            output_directory=tmp_path / "missing-output",
        )

    ambiguous_acquisition, ambiguous_catalog, ambiguous_selection = _fixture(
        tmp_path / "ambiguous", monkeypatch, duplicate_standard_work=True
    )
    with pytest.raises(ValueError, match="ambiguous"):
        build_corpus(
            acquisition_directory=ambiguous_acquisition,
            catalog_path=ambiguous_catalog,
            selection_config_path=ambiguous_selection,
            output_directory=tmp_path / "ambiguous-output",
        )


def test_corpus_cli_handlers_forward_exact_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, object] = {}

    def acquire(config_path: Path, destination: Path) -> object:
        calls["acquire"] = (config_path, destination)
        return object()

    def build(**arguments: Path) -> object:
        calls["build"] = arguments
        return object()

    monkeypatch.setattr("poetry50m.data.hf_sources.acquire_hf_sources", acquire)
    monkeypatch.setattr("poetry50m.data.corpus_builder.build_corpus", build)
    sources_config = tmp_path / "sources.json"
    acquisition = tmp_path / "acquired"
    catalog = acquisition / "acquisition_receipt.json"
    selection = tmp_path / "selection.json"
    output = tmp_path / "corpus"

    assert (
        main(
            (
                "corpus-acquire",
                "--sources-config",
                str(sources_config),
                "--output",
                str(acquisition),
            )
        )
        == 0
    )
    assert (
        main(
            (
                "corpus-build",
                "--acquisition",
                str(acquisition),
                "--catalog",
                str(catalog),
                "--selection-config",
                str(selection),
                "--output",
                str(output),
            )
        )
        == 0
    )
    assert calls["acquire"] == (sources_config, acquisition)
    assert calls["build"] == {
        "acquisition_directory": acquisition,
        "catalog_path": catalog,
        "selection_config_path": selection,
        "output_directory": output,
    }
