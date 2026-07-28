from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from huggingface_hub import HfApi

from poetry50m.config import canonical_json
from poetry50m.data.hf_sources import (
    AcquisitionReceipt,
    ArtifactSpec,
    HfSourcesConfig,
    HfSourceSpec,
    HubDownloader,
    acquire_hf_sources,
    load_hf_sources_config,
    verify_acquisition,
)

PROJECT_ROOT = Path(__file__).parents[2]
CATALOG_PATH = PROJECT_ROOT / "configs/data/huggingface_sources.json"


def test_catalog_contains_only_the_frozen_sources_and_raw_artifacts() -> None:
    config = load_hf_sources_config(CATALOG_PATH)

    assert tuple((source.repository, source.revision) for source in config.sources) == (
        (
            "openbmb/Ultra-FineWeb-L3",
            "c68ab81ad03b2d2f476fa8ab3c72bed3528da359",
        ),
        (
            "biglam/gutenberg-poetry-corpus",
            "fcd42e249fed48dbd1d3b9b969528ef9298d3464",
        ),
        (
            "yoonholee/poetry-greats-public-domain",
            "3201e250462905a0c8f6134e124382ac96586dc9",
        ),
    )
    assert tuple(artifact.path for source in config.sources for artifact in source.artifacts) == (
        "data/ultrafineweb_en_l3/multi_style/part-00000-f36f5a53-4a77-434b-b9bc-67ed69b93fe2-c000.snappy.parquet",
        "data/train-00000-of-00001-fa9fb9e1f16eed7e.parquet",
        "data/train-00000-of-00001.parquet",
    )
    assert tuple(source.repository_files for source in config.sources) == (
        (
            ".gitattributes",
            "LICENSE",
            "README.md",
            "data/ultrafineweb_en_l3/multi_style/part-00000-f36f5a53-4a77-434b-b9bc-67ed69b93fe2-c000.snappy.parquet",
        ),
        (
            ".gitattributes",
            "README.md",
            "dataset_infos.json",
            "data/train-00000-of-00001-fa9fb9e1f16eed7e.parquet",
        ),
        (
            ".gitattributes",
            "README.md",
            "data/train-00000-of-00001.parquet",
        ),
    )


def test_catalog_rejects_mutable_revisions_unknown_fields_and_identity_drift(
    tmp_path: Path,
) -> None:
    raw = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))

    raw["sources"][0]["revision"] = "main"
    mutable_path = tmp_path / "mutable.json"
    mutable_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="40-character"):
        load_hf_sources_config(mutable_path)

    raw = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    raw["unexpected"] = True
    unexpected_path = tmp_path / "unexpected.json"
    unexpected_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly"):
        load_hf_sources_config(unexpected_path)

    raw = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    raw["sources"][0]["revision"] = "0" * 40
    drifted_path = tmp_path / "drifted.json"
    drifted_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen approved catalog"):
        load_hf_sources_config(drifted_path)


class FakeHubApi:
    def __init__(
        self,
        config: HfSourcesConfig,
        *,
        resolved_revision: str | None = None,
        extra_repository_file: str | None = None,
        missing_repository_file: str | None = None,
    ) -> None:
        self._config = config
        self._resolved_revision = resolved_revision
        self._extra_repository_file = extra_repository_file
        self._missing_repository_file = missing_repository_file
        self.calls: list[tuple[str, str, bool]] = []

    def dataset_info(
        self,
        repo_id: str,
        *,
        revision: str,
        files_metadata: bool,
    ) -> SimpleNamespace:
        self.calls.append((repo_id, revision, files_metadata))
        source = next(item for item in self._config.sources if item.repository == repo_id)
        siblings = [
            SimpleNamespace(rfilename=filename)
            for filename in source.repository_files
            if filename != self._missing_repository_file
        ]
        if self._extra_repository_file is not None:
            siblings.append(SimpleNamespace(rfilename=self._extra_repository_file))
        return SimpleNamespace(
            sha=self._resolved_revision or revision,
            siblings=siblings,
        )


class FakeDownloader:
    def __init__(self, cache: Path, content: dict[tuple[str, str], bytes]) -> None:
        self._cache = cache
        self._content = content
        self.calls: list[dict[str, str | None]] = []

    def __call__(
        self,
        *,
        repo_id: str,
        filename: str,
        repo_type: str,
        revision: str,
        cache_dir: str | None,
    ) -> str:
        self.calls.append(
            {
                "repo_id": repo_id,
                "filename": filename,
                "repo_type": repo_type,
                "revision": revision,
                "cache_dir": cache_dir,
            }
        )
        path = self._cache / repo_id.replace("/", "--") / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self._content[(repo_id, filename)])
        return str(path)


def _small_config() -> tuple[HfSourcesConfig, dict[tuple[str, str], bytes]]:
    approved = load_hf_sources_config(CATALOG_PATH)
    content: dict[tuple[str, str], bytes] = {}
    sources: list[HfSourceSpec] = []
    for source in approved.sources:
        artifacts: list[ArtifactSpec] = []
        for index, artifact in enumerate(source.artifacts):
            payload = f"{source.source_id}:{index}\n".encode()
            content[(source.repository, artifact.path)] = payload
            artifacts.append(
                ArtifactSpec(
                    path=artifact.path,
                    sha256=hashlib.sha256(payload).hexdigest(),
                    size_bytes=len(payload),
                )
            )
        sources.append(replace(source, artifacts=tuple(artifacts)))
    return HfSourcesConfig(format_version=1, sources=tuple(sources)), content


def _install_small_config(monkeypatch: pytest.MonkeyPatch, config: HfSourcesConfig) -> None:
    monkeypatch.setattr(
        "poetry50m.data.hf_sources.load_hf_sources_config",
        lambda _path: config,
    )


def test_acquisition_uses_dataset_api_pins_and_writes_verified_canonical_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, content = _small_config()
    _install_small_config(monkeypatch, config)
    api = FakeHubApi(config)
    downloader = FakeDownloader(tmp_path / "hub-cache", content)
    destination = tmp_path / "raw"
    cache_directory = tmp_path / "specified-cache"

    receipt = acquire_hf_sources(
        CATALOG_PATH,
        destination,
        cache_directory=cache_directory,
        api=cast(HfApi, api),
        downloader=cast(HubDownloader, downloader),
    )

    assert isinstance(receipt, AcquisitionReceipt)
    assert receipt.config_sha256 == config.sha256
    assert api.calls == [(source.repository, source.revision, True) for source in config.sources]
    assert {
        (call["repo_id"], call["filename"], call["repo_type"], call["revision"])
        for call in downloader.calls
    } == {
        (source.repository, artifact.path, "dataset", source.revision)
        for source in config.sources
        for artifact in source.artifacts
    }
    assert {call["cache_dir"] for call in downloader.calls} == {str(cache_directory)}
    receipt_path = destination / "acquisition_receipt.json"
    assert receipt_path.read_text(encoding="utf-8") == (canonical_json(receipt.to_mapping()) + "\n")
    assert verify_acquisition(CATALOG_PATH, destination) == receipt


def test_acquisition_rejects_remote_revision_and_missing_required_repository_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, content = _small_config()
    _install_small_config(monkeypatch, config)
    downloader = FakeDownloader(tmp_path / "cache", content)

    with pytest.raises(ValueError, match="resolved to"):
        acquire_hf_sources(
            CATALOG_PATH,
            tmp_path / "wrong-revision",
            api=cast(HfApi, FakeHubApi(config, resolved_revision="f" * 40)),
            downloader=cast(HubDownloader, downloader),
        )
    assert not (tmp_path / "wrong-revision").exists()

    api = FakeHubApi(config, missing_repository_file=config.sources[0].repository_files[0])
    with pytest.raises(ValueError, match="missing required inspected files"):
        acquire_hf_sources(
            CATALOG_PATH,
            tmp_path / "missing-required-file",
            api=cast(HfApi, api),
            downloader=cast(HubDownloader, downloader),
        )
    assert not (tmp_path / "missing-required-file").exists()


def test_acquisition_allows_unrelated_files_at_the_pinned_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, content = _small_config()
    _install_small_config(monkeypatch, config)

    receipt = acquire_hf_sources(
        CATALOG_PATH,
        tmp_path / "raw",
        api=cast(HfApi, FakeHubApi(config, extra_repository_file="unrelated.bin")),
        downloader=cast(HubDownloader, FakeDownloader(tmp_path / "cache", content)),
    )

    assert receipt.sources[0].source_id == "ultrafineweb_l3"


def test_acquisition_rejects_bad_download_and_cleans_staging_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, content = _small_config()
    _install_small_config(monkeypatch, config)
    first_key = next(iter(content))
    content[first_key] = b"tampered"
    destination = tmp_path / "raw"

    with pytest.raises(ValueError, match="mismatch"):
        acquire_hf_sources(
            CATALOG_PATH,
            destination,
            api=cast(HfApi, FakeHubApi(config)),
            downloader=cast(HubDownloader, FakeDownloader(tmp_path / "cache", content)),
        )

    assert not destination.exists()
    assert not tuple(tmp_path.glob(".raw.staging-*"))


def test_verifier_rejects_receipt_config_and_local_layout_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, content = _small_config()
    _install_small_config(monkeypatch, config)
    destination = tmp_path / "raw"
    acquire_hf_sources(
        CATALOG_PATH,
        destination,
        api=cast(HfApi, FakeHubApi(config)),
        downloader=cast(HubDownloader, FakeDownloader(tmp_path / "cache", content)),
    )

    unexpected = destination / "unexpected.txt"
    unexpected.write_text("drift", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected_files"):
        verify_acquisition(CATALOG_PATH, destination)
    unexpected.unlink()

    receipt_path = destination / "acquisition_receipt.json"
    value = json.loads(receipt_path.read_text(encoding="utf-8"))
    value["config_sha256"] = "0" * 64
    receipt_path.write_text(canonical_json(value) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="config hash"):
        verify_acquisition(CATALOG_PATH, destination)
