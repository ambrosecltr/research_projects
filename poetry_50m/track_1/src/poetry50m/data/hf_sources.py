"""Pinned, fail-closed acquisition of approved Hugging Face dataset artifacts."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol, cast

from huggingface_hub import HfApi, hf_hub_download

from poetry50m.config import canonical_json, file_hash

_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_SOURCE_ID_PATTERN = re.compile(r"[a-z][a-z0-9_]*")
_ARTIFACT_KINDS = {
    "distilled_text_jsonl",
    "synthetic_knowledge_jsonl",
}
_APPROVED_CONFIG_SHA256 = "e99314114c7a4a8f22a259cbe8bcc979288165731665cb41def3bfd7b65c692f"
_RECEIPT_NAME = "acquisition_receipt.json"


def _required_string(name: str, value: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")


def _validate_relative_path(name: str, value: str) -> None:
    _required_string(name, value)
    path = PurePosixPath(value)
    if (
        "\\" in value
        or path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{name} must be a normalized relative POSIX path")


def _exact_mapping(value: object, *, name: str, keys: set[str]) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a JSON object")
    raw = cast(dict[object, object], value)
    if any(not isinstance(key, str) for key in raw):
        raise TypeError(f"{name} keys must be strings")
    mapping = {cast(str, key): item for key, item in raw.items()}
    if set(mapping) != keys:
        raise ValueError(f"{name} must contain exactly {sorted(keys)}")
    return mapping


def _string_field(mapping: dict[str, object], field: str, *, owner: str) -> str:
    value = mapping[field]
    if not isinstance(value, str) or not value:
        raise TypeError(f"{owner}.{field} must be a non-empty string")
    return value


def _integer_field(mapping: dict[str, object], field: str, *, owner: str) -> int:
    value = mapping[field]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{owner}.{field} must be an integer")
    return value


def _object_list(mapping: dict[str, object], field: str, *, owner: str) -> list[object]:
    value = mapping[field]
    if not isinstance(value, list):
        raise TypeError(f"{owner}.{field} must be a JSON list")
    return cast(list[object], value)


def _string_list(mapping: dict[str, object], field: str, *, owner: str) -> tuple[str, ...]:
    values = _object_list(mapping, field, owner=owner)
    if any(not isinstance(item, str) for item in values):
        raise TypeError(f"{owner}.{field} must contain only strings")
    return tuple(cast(str, item) for item in values)


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_json(text: str, *, name: str) -> object:
    try:
        return json.loads(text, object_pairs_hook=_unique_json_object)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {name}: {error}") from error


@dataclass(frozen=True, slots=True)
class ArtifactSpec:
    """Expected identity of one raw artifact in a pinned Hub repository."""

    path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        _validate_relative_path("artifact.path", self.path)
        if not isinstance(self.sha256, str) or not _SHA256_PATTERN.fullmatch(self.sha256):
            raise ValueError("artifact.sha256 must be a lowercase SHA-256")
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes < 1
        ):
            raise ValueError("artifact.size_bytes must be a positive integer")

    @classmethod
    def from_mapping(cls, value: object) -> ArtifactSpec:
        mapping = _exact_mapping(
            value,
            name="artifact",
            keys={"path", "sha256", "size_bytes"},
        )
        return cls(
            path=_string_field(mapping, "path", owner="artifact"),
            sha256=_string_field(mapping, "sha256", owner="artifact"),
            size_bytes=_integer_field(mapping, "size_bytes", owner="artifact"),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class HfSourceSpec:
    """One approved dataset repository and its complete raw-file contract."""

    source_id: str
    repository: str
    revision: str
    artifact_kind: str
    repository_files: tuple[str, ...]
    artifacts: tuple[ArtifactSpec, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, str) or not _SOURCE_ID_PATTERN.fullmatch(self.source_id):
            raise ValueError("source_id must be lowercase snake case")
        _required_string("repository", self.repository)
        if self.repository.count("/") != 1 or any(
            not component for component in self.repository.split("/")
        ):
            raise ValueError("repository must be a Hugging Face owner/name identifier")
        if not isinstance(self.revision, str) or not _COMMIT_PATTERN.fullmatch(self.revision):
            raise ValueError("revision must be a full lowercase 40-character commit SHA")
        if self.artifact_kind not in _ARTIFACT_KINDS:
            raise ValueError(f"unsupported artifact_kind: {self.artifact_kind}")
        if not isinstance(self.repository_files, tuple) or not self.repository_files:
            raise TypeError("repository_files must be a non-empty tuple")
        for path in self.repository_files:
            _validate_relative_path("repository_files entry", path)
        if len(self.repository_files) != len(set(self.repository_files)):
            raise ValueError("repository_files must not contain duplicates")
        if not isinstance(self.artifacts, tuple) or not self.artifacts:
            raise TypeError("artifacts must be a non-empty tuple")
        if any(not isinstance(artifact, ArtifactSpec) for artifact in self.artifacts):
            raise TypeError("artifacts must contain only ArtifactSpec records")
        artifact_paths = tuple(artifact.path for artifact in self.artifacts)
        if len(artifact_paths) != len(set(artifact_paths)):
            raise ValueError("artifacts must not contain duplicate paths")
        missing = set(artifact_paths).difference(self.repository_files)
        if missing:
            raise ValueError(f"artifacts are absent from repository_files: {sorted(missing)}")

    @classmethod
    def from_mapping(cls, value: object) -> HfSourceSpec:
        mapping = _exact_mapping(
            value,
            name="Hugging Face source",
            keys={
                "artifact_kind",
                "artifacts",
                "repository",
                "repository_files",
                "revision",
                "source_id",
            },
        )
        return cls(
            source_id=_string_field(mapping, "source_id", owner="source"),
            repository=_string_field(mapping, "repository", owner="source"),
            revision=_string_field(mapping, "revision", owner="source"),
            artifact_kind=_string_field(mapping, "artifact_kind", owner="source"),
            repository_files=_string_list(mapping, "repository_files", owner="source"),
            artifacts=tuple(
                ArtifactSpec.from_mapping(item)
                for item in _object_list(mapping, "artifacts", owner="source")
            ),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "repository": self.repository,
            "revision": self.revision,
            "artifact_kind": self.artifact_kind,
            "repository_files": list(self.repository_files),
            "artifacts": [artifact.to_mapping() for artifact in self.artifacts],
        }


@dataclass(frozen=True, slots=True)
class HfSourcesConfig:
    """Validated acquisition catalog for the two approved knowledge sources."""

    format_version: int
    sources: tuple[HfSourceSpec, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.format_version, bool)
            or not isinstance(self.format_version, int)
            or self.format_version != 1
        ):
            raise ValueError("Hugging Face source format_version must be 1")
        if not isinstance(self.sources, tuple) or len(self.sources) != 2:
            raise ValueError("Hugging Face source catalog must contain exactly two sources")
        if any(not isinstance(source, HfSourceSpec) for source in self.sources):
            raise TypeError("sources must contain only HfSourceSpec records")
        source_ids = tuple(source.source_id for source in self.sources)
        repositories = tuple(source.repository for source in self.sources)
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("source IDs must be unique")
        if len(repositories) != len(set(repositories)):
            raise ValueError("source repositories must be unique")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json(self.to_mapping()).encode("utf-8")).hexdigest()

    @classmethod
    def from_mapping(cls, value: object) -> HfSourcesConfig:
        mapping = _exact_mapping(
            value,
            name="Hugging Face source catalog",
            keys={"format_version", "sources"},
        )
        return cls(
            format_version=_integer_field(mapping, "format_version", owner="catalog"),
            sources=tuple(
                HfSourceSpec.from_mapping(item)
                for item in _object_list(mapping, "sources", owner="catalog")
            ),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "sources": [source.to_mapping() for source in self.sources],
        }


def load_hf_sources_config(path: Path) -> HfSourcesConfig:
    """Load the catalog and reject any identity, artifact, or schema drift."""
    if not path.is_file():
        raise ValueError(f"expected a Hugging Face source catalog file: {path}")
    config = HfSourcesConfig.from_mapping(
        _parse_json(path.read_text(encoding="utf-8"), name=str(path))
    )
    if config.sha256 != _APPROVED_CONFIG_SHA256:
        raise ValueError("Hugging Face source catalog differs from the frozen approved catalog")
    return config


@dataclass(frozen=True, slots=True)
class AcquiredArtifact:
    """Verified local identity of one acquired raw artifact."""

    source_path: str
    local_path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        _validate_relative_path("source_path", self.source_path)
        _validate_relative_path("local_path", self.local_path)
        if not isinstance(self.sha256, str) or not _SHA256_PATTERN.fullmatch(self.sha256):
            raise ValueError("acquired artifact sha256 must be a lowercase SHA-256")
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes < 1
        ):
            raise ValueError("acquired artifact size_bytes must be a positive integer")

    @classmethod
    def from_mapping(cls, value: object) -> AcquiredArtifact:
        mapping = _exact_mapping(
            value,
            name="acquired artifact",
            keys={"local_path", "sha256", "size_bytes", "source_path"},
        )
        return cls(
            source_path=_string_field(mapping, "source_path", owner="acquired artifact"),
            local_path=_string_field(mapping, "local_path", owner="acquired artifact"),
            sha256=_string_field(mapping, "sha256", owner="acquired artifact"),
            size_bytes=_integer_field(mapping, "size_bytes", owner="acquired artifact"),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "source_path": self.source_path,
            "local_path": self.local_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class AcquiredSource:
    """Receipt entry binding local artifacts to one resolved Hub commit."""

    source_id: str
    repository: str
    revision: str
    resolved_revision: str
    artifact_kind: str
    artifacts: tuple[AcquiredArtifact, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, str) or not _SOURCE_ID_PATTERN.fullmatch(self.source_id):
            raise ValueError("acquired source_id must be lowercase snake case")
        _required_string("repository", self.repository)
        for name, revision in (
            ("revision", self.revision),
            ("resolved_revision", self.resolved_revision),
        ):
            if not isinstance(revision, str) or not _COMMIT_PATTERN.fullmatch(revision):
                raise ValueError(f"{name} must be a full lowercase 40-character commit SHA")
        if self.artifact_kind not in _ARTIFACT_KINDS:
            raise ValueError(f"unsupported acquired artifact_kind: {self.artifact_kind}")
        if not isinstance(self.artifacts, tuple) or not self.artifacts:
            raise TypeError("acquired artifacts must be a non-empty tuple")
        if any(not isinstance(artifact, AcquiredArtifact) for artifact in self.artifacts):
            raise TypeError("acquired artifacts must contain only AcquiredArtifact records")

    @classmethod
    def from_mapping(cls, value: object) -> AcquiredSource:
        mapping = _exact_mapping(
            value,
            name="acquired source",
            keys={
                "artifact_kind",
                "artifacts",
                "repository",
                "resolved_revision",
                "revision",
                "source_id",
            },
        )
        return cls(
            source_id=_string_field(mapping, "source_id", owner="acquired source"),
            repository=_string_field(mapping, "repository", owner="acquired source"),
            revision=_string_field(mapping, "revision", owner="acquired source"),
            resolved_revision=_string_field(mapping, "resolved_revision", owner="acquired source"),
            artifact_kind=_string_field(mapping, "artifact_kind", owner="acquired source"),
            artifacts=tuple(
                AcquiredArtifact.from_mapping(item)
                for item in _object_list(mapping, "artifacts", owner="acquired source")
            ),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "repository": self.repository,
            "revision": self.revision,
            "resolved_revision": self.resolved_revision,
            "artifact_kind": self.artifact_kind,
            "artifacts": [artifact.to_mapping() for artifact in self.artifacts],
        }


@dataclass(frozen=True, slots=True)
class AcquisitionReceipt:
    """Canonical receipt for a complete, locally verified acquisition."""

    format_version: int
    config_sha256: str
    sources: tuple[AcquiredSource, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.format_version, bool)
            or not isinstance(self.format_version, int)
            or self.format_version != 1
        ):
            raise ValueError("acquisition receipt format_version must be 1")
        if not isinstance(self.config_sha256, str) or not _SHA256_PATTERN.fullmatch(
            self.config_sha256
        ):
            raise ValueError("acquisition receipt config_sha256 must be a lowercase SHA-256")
        if not isinstance(self.sources, tuple) or len(self.sources) != 2:
            raise ValueError("acquisition receipt must contain exactly two sources")
        if any(not isinstance(source, AcquiredSource) for source in self.sources):
            raise TypeError("receipt sources must contain only AcquiredSource records")

    @classmethod
    def from_mapping(cls, value: object) -> AcquisitionReceipt:
        mapping = _exact_mapping(
            value,
            name="acquisition receipt",
            keys={"config_sha256", "format_version", "sources"},
        )
        return cls(
            format_version=_integer_field(mapping, "format_version", owner="receipt"),
            config_sha256=_string_field(mapping, "config_sha256", owner="receipt"),
            sources=tuple(
                AcquiredSource.from_mapping(item)
                for item in _object_list(mapping, "sources", owner="receipt")
            ),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "config_sha256": self.config_sha256,
            "sources": [source.to_mapping() for source in self.sources],
        }


class HubDownloader(Protocol):
    def __call__(
        self,
        *,
        repo_id: str,
        filename: str,
        repo_type: str,
        revision: str,
        cache_dir: str | None,
    ) -> str: ...


def _verify_remote_repository(api: HfApi, source: HfSourceSpec) -> str:
    info = api.dataset_info(
        repo_id=source.repository,
        revision=source.revision,
        files_metadata=True,
    )
    if info.sha != source.revision:
        raise ValueError(
            f"{source.repository} resolved to {info.sha!r}, expected {source.revision}"
        )
    if info.siblings is None:
        raise ValueError(f"{source.repository} did not return a repository file listing")
    remote_files = tuple(sibling.rfilename for sibling in info.siblings)
    if len(remote_files) != len(set(remote_files)):
        raise ValueError(f"{source.repository} returned duplicate repository filenames")
    missing = set(source.repository_files).difference(remote_files)
    unexpected = set(remote_files).difference(source.repository_files)
    if missing or unexpected:
        raise ValueError(
            f"{source.repository} repository files drifted: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    return info.sha


def _local_artifact_path(source: HfSourceSpec, artifact: ArtifactSpec) -> str:
    return f"{source.source_id}/{artifact.path}"


def _verify_file(path: Path, artifact: ArtifactSpec) -> tuple[str, int]:
    if not path.is_file():
        raise ValueError(f"expected a downloaded regular file: {path}")
    size_bytes = path.stat().st_size
    if size_bytes != artifact.size_bytes:
        raise ValueError(
            f"artifact size mismatch for {artifact.path}: "
            f"expected {artifact.size_bytes}, got {size_bytes}"
        )
    digest = file_hash(path)
    if digest != artifact.sha256:
        raise ValueError(
            f"artifact SHA-256 mismatch for {artifact.path}: "
            f"expected {artifact.sha256}, got {digest}"
        )
    return digest, size_bytes


def _expected_local_layout(config: HfSourcesConfig) -> tuple[set[str], set[str]]:
    files = {_RECEIPT_NAME}
    directories: set[str] = set()
    for source in config.sources:
        for artifact in source.artifacts:
            local_path = PurePosixPath(_local_artifact_path(source, artifact))
            files.add(local_path.as_posix())
            directories.update(
                parent.as_posix() for parent in local_path.parents if parent.as_posix() != "."
            )
    return files, directories


def _assert_exact_local_layout(destination: Path, config: HfSourcesConfig) -> None:
    if destination.is_symlink() or not destination.is_dir():
        raise ValueError(f"acquisition destination must be a real directory: {destination}")
    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    for path in destination.rglob("*"):
        relative = path.relative_to(destination).as_posix()
        if path.is_symlink():
            raise ValueError(f"acquisition contains a symbolic link: {relative}")
        if path.is_file():
            actual_files.add(relative)
        elif path.is_dir():
            actual_directories.add(relative)
        else:
            raise ValueError(f"acquisition contains an unsupported filesystem entry: {relative}")
    expected_files, expected_directories = _expected_local_layout(config)
    if actual_files != expected_files or actual_directories != expected_directories:
        raise ValueError(
            "acquisition layout drifted: "
            f"missing_files={sorted(expected_files - actual_files)}, "
            f"unexpected_files={sorted(actual_files - expected_files)}, "
            f"missing_directories={sorted(expected_directories - actual_directories)}, "
            f"unexpected_directories={sorted(actual_directories - expected_directories)}"
        )


def _validate_receipt(receipt: AcquisitionReceipt, config: HfSourcesConfig) -> None:
    if receipt.config_sha256 != config.sha256:
        raise ValueError("acquisition receipt config hash does not match the source catalog")
    if len(receipt.sources) != len(config.sources):
        raise ValueError("acquisition receipt source count differs from the source catalog")
    for expected_source, acquired_source in zip(config.sources, receipt.sources, strict=True):
        expected_identity = (
            expected_source.source_id,
            expected_source.repository,
            expected_source.revision,
            expected_source.revision,
            expected_source.artifact_kind,
        )
        acquired_identity = (
            acquired_source.source_id,
            acquired_source.repository,
            acquired_source.revision,
            acquired_source.resolved_revision,
            acquired_source.artifact_kind,
        )
        if acquired_identity != expected_identity:
            raise ValueError(f"acquisition receipt source drift for {expected_source.source_id}")
        if len(acquired_source.artifacts) != len(expected_source.artifacts):
            raise ValueError(
                f"acquisition receipt artifact count drift for {expected_source.source_id}"
            )
        for expected_artifact, acquired_artifact in zip(
            expected_source.artifacts, acquired_source.artifacts, strict=True
        ):
            expected_artifact_identity = (
                expected_artifact.path,
                _local_artifact_path(expected_source, expected_artifact),
                expected_artifact.sha256,
                expected_artifact.size_bytes,
            )
            acquired_artifact_identity = (
                acquired_artifact.source_path,
                acquired_artifact.local_path,
                acquired_artifact.sha256,
                acquired_artifact.size_bytes,
            )
            if acquired_artifact_identity != expected_artifact_identity:
                raise ValueError(f"acquisition receipt artifact drift for {expected_artifact.path}")


def _receipt_text(receipt: AcquisitionReceipt) -> str:
    return canonical_json(receipt.to_mapping()) + "\n"


def verify_acquisition(config_path: Path, destination: Path) -> AcquisitionReceipt:
    """Verify a complete acquisition without performing network access."""
    config = load_hf_sources_config(config_path)
    _assert_exact_local_layout(destination, config)
    receipt_path = destination / _RECEIPT_NAME
    receipt_text = receipt_path.read_text(encoding="utf-8")
    receipt = AcquisitionReceipt.from_mapping(_parse_json(receipt_text, name=str(receipt_path)))
    if receipt_text != _receipt_text(receipt):
        raise ValueError("acquisition receipt is not canonical JSON")
    _validate_receipt(receipt, config)
    for source, acquired_source in zip(config.sources, receipt.sources, strict=True):
        for artifact, acquired_artifact in zip(
            source.artifacts, acquired_source.artifacts, strict=True
        ):
            _verify_file(destination / acquired_artifact.local_path, artifact)
    return receipt


def _build_acquisition(
    *,
    config: HfSourcesConfig,
    staging: Path,
    cache_directory: Path | None,
    api: HfApi,
    downloader: HubDownloader,
) -> AcquisitionReceipt:
    acquired_sources: list[AcquiredSource] = []
    for source in config.sources:
        resolved_revision = _verify_remote_repository(api, source)
        acquired_artifacts: list[AcquiredArtifact] = []
        for artifact in source.artifacts:
            cached_path = Path(
                downloader(
                    repo_id=source.repository,
                    filename=artifact.path,
                    repo_type="dataset",
                    revision=source.revision,
                    cache_dir=str(cache_directory) if cache_directory is not None else None,
                )
            )
            if not cached_path.is_file():
                raise ValueError(
                    f"Hugging Face downloader returned a non-file for {artifact.path}: "
                    f"{cached_path}"
                )
            local_path = _local_artifact_path(source, artifact)
            target = staging / local_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cached_path, target)
            digest, size_bytes = _verify_file(target, artifact)
            acquired_artifacts.append(
                AcquiredArtifact(
                    source_path=artifact.path,
                    local_path=local_path,
                    sha256=digest,
                    size_bytes=size_bytes,
                )
            )
        acquired_sources.append(
            AcquiredSource(
                source_id=source.source_id,
                repository=source.repository,
                revision=source.revision,
                resolved_revision=resolved_revision,
                artifact_kind=source.artifact_kind,
                artifacts=tuple(acquired_artifacts),
            )
        )
    receipt = AcquisitionReceipt(
        format_version=1,
        config_sha256=config.sha256,
        sources=tuple(acquired_sources),
    )
    (staging / _RECEIPT_NAME).write_text(_receipt_text(receipt), encoding="utf-8", newline="\n")
    return receipt


def acquire_hf_sources(
    config_path: Path,
    destination: Path,
    *,
    cache_directory: Path | None = None,
    api: HfApi | None = None,
    downloader: HubDownloader | None = None,
) -> AcquisitionReceipt:
    """Acquire the frozen catalog into a new directory and return its verified receipt."""
    config = load_hf_sources_config(config_path)
    if destination.is_symlink():
        raise ValueError(f"acquisition destination must not be a symbolic link: {destination}")
    if destination.exists():
        return verify_acquisition(config_path, destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-",
            dir=destination.parent,
        )
    )
    try:
        hub_api = api if api is not None else HfApi()
        hub_downloader = (
            downloader if downloader is not None else cast(HubDownloader, hf_hub_download)
        )
        _build_acquisition(
            config=config,
            staging=staging,
            cache_directory=cache_directory,
            api=hub_api,
            downloader=hub_downloader,
        )
        _assert_exact_local_layout(staging, config)
        staging.rename(destination)
        return verify_acquisition(config_path, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
