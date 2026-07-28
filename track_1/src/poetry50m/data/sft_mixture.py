"""Pinned acquisition and exact-token assembly for the Track 1 SFT mixture."""

from __future__ import annotations

import hashlib
import heapq
import json
import shutil
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, cast

import pyarrow.parquet as pq  # type: ignore[import-untyped]
from huggingface_hub import HfApi, hf_hub_download
from tokenizers import Tokenizer

from poetry50m.config import file_hash

from .artifacts import write_packed_sequences
from .packing import pack_sequences
from .schema import TokenSequence
from .splits import LexicalFamilyIndex
from .synthetic_sft import DEFAULT_TARGET_TOKENS, TRACK1_SFT_TOKENIZER_SHA256
from .tokenizer import load_tokenizer

FORMAT_VERSION = 1
SELECTION_SEED = "poetry50m-sft-mixture-v1"
SEQUENCE_LENGTH = 1024
CANDIDATE_POOL_SIZE = 120_000


@dataclass(frozen=True, slots=True)
class SourceArtifact:
    split: Literal["train", "test"]
    path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class SftSource:
    source_id: str
    repository: str
    revision: str
    licence: str
    artifacts: tuple[SourceArtifact, ...]


@dataclass(frozen=True, slots=True)
class SmolCandidate:
    sort_key: str
    example_id: str
    artifact_path: str
    row_index: int
    formatted_tokens: int
    supervised_tokens: int
    conversation_hash: str
    assistant_hashes: tuple[str, ...]


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_json(path: Path, value: object) -> None:
    path.write_text(_canonical_json(value) + "\n", encoding="utf-8", newline="\n")


def _write_jsonl(path: Path, values: Iterable[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(_canonical_json(value) + "\n")


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{path} must contain a JSON object")
    return cast(dict[str, object], value)


def _required_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _required_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"{name} must be a non-negative integer")
    return value


def _load_source_config(path: Path) -> tuple[str, SftSource]:
    raw = _read_json(path)
    if set(raw) != {"format_version", "sources"} or raw["format_version"] != FORMAT_VERSION:
        raise ValueError("unsupported SFT source config")
    sources = raw["sources"]
    if not isinstance(sources, list) or len(sources) != 1 or not isinstance(sources[0], dict):
        raise ValueError("SFT source config must contain exactly one source")
    source_raw = cast(dict[str, object], sources[0])
    if set(source_raw) != {
        "source_id",
        "repository",
        "revision",
        "licence",
        "artifacts",
    }:
        raise ValueError("SFT source entry has unexpected fields")
    artifact_values = source_raw["artifacts"]
    if not isinstance(artifact_values, list) or not artifact_values:
        raise ValueError("SFT source requires artifacts")
    artifacts: list[SourceArtifact] = []
    for raw_artifact in artifact_values:
        if not isinstance(raw_artifact, dict) or set(raw_artifact) != {
            "split",
            "path",
            "sha256",
            "size_bytes",
        }:
            raise ValueError("SFT source artifact has unexpected fields")
        split = raw_artifact["split"]
        if split not in {"train", "test"}:
            raise ValueError("SFT source artifact split must be train or test")
        artifact_path = _required_string(raw_artifact["path"], name="artifact path")
        pure_path = PurePosixPath(artifact_path)
        if pure_path.is_absolute() or ".." in pure_path.parts:
            raise ValueError("artifact path must be relative")
        digest = _required_string(raw_artifact["sha256"], name="artifact sha256")
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("artifact sha256 must be lowercase SHA-256")
        artifacts.append(
            SourceArtifact(
                split=cast(Literal["train", "test"], split),
                path=artifact_path,
                sha256=digest,
                size_bytes=_required_integer(
                    raw_artifact["size_bytes"], name="artifact size_bytes"
                ),
            )
        )
    source = SftSource(
        source_id=_required_string(source_raw["source_id"], name="source_id"),
        repository=_required_string(source_raw["repository"], name="repository"),
        revision=_required_string(source_raw["revision"], name="revision"),
        licence=_required_string(source_raw["licence"], name="licence"),
        artifacts=tuple(artifacts),
    )
    if len(source.revision) != 40:
        raise ValueError("source revision must be a full commit SHA")
    return file_hash(path), source


def _verify_artifact(path: Path, artifact: SourceArtifact) -> None:
    if not path.is_file():
        raise ValueError(f"missing SFT source artifact: {path}")
    if path.stat().st_size != artifact.size_bytes:
        raise ValueError(f"size mismatch for {artifact.path}")
    if file_hash(path) != artifact.sha256:
        raise ValueError(f"SHA-256 mismatch for {artifact.path}")


def verify_smoltalk_acquisition(config_path: Path, output_directory: Path) -> dict[str, object]:
    config_sha256, source = _load_source_config(config_path)
    receipt = _read_json(output_directory / "receipt.json")
    if receipt.get("config_sha256") != config_sha256:
        raise ValueError("SmolTalk acquisition config hash mismatch")
    if receipt.get("revision") != source.revision:
        raise ValueError("SmolTalk acquisition revision mismatch")
    for artifact in source.artifacts:
        _verify_artifact(output_directory / artifact.path, artifact)
    return receipt


def acquire_smoltalk(config_path: Path, output_directory: Path) -> Path:
    """Download and verify the frozen Smol-Smoltalk train and test Parquet files."""
    config_sha256, source = _load_source_config(config_path)
    if output_directory.exists():
        verify_smoltalk_acquisition(config_path, output_directory)
        return output_directory / "receipt.json"
    info = HfApi().dataset_info(
        source.repository,
        revision=source.revision,
        files_metadata=True,
    )
    if info.sha != source.revision:
        raise ValueError("SmolTalk repository did not resolve to its frozen revision")
    remote_files = {sibling.rfilename for sibling in info.siblings or ()}
    missing = {artifact.path for artifact in source.artifacts}.difference(remote_files)
    if missing:
        raise ValueError(f"SmolTalk repository is missing files: {sorted(missing)}")
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_directory.name}.staging-",
            dir=output_directory.parent,
        )
    )
    try:
        for artifact in source.artifacts:
            cached = Path(
                hf_hub_download(
                    repo_id=source.repository,
                    repo_type="dataset",
                    revision=source.revision,
                    filename=artifact.path,
                )
            )
            destination = staging / artifact.path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cached, destination)
            _verify_artifact(destination, artifact)
        _write_json(
            staging / "receipt.json",
            {
                "format_version": FORMAT_VERSION,
                "config_sha256": config_sha256,
                "source_id": source.source_id,
                "repository": source.repository,
                "revision": source.revision,
                "licence": source.licence,
                "artifacts": [asdict(artifact) for artifact in source.artifacts],
            },
        )
        staging.rename(output_directory)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    verify_smoltalk_acquisition(config_path, output_directory)
    return output_directory / "receipt.json"


def _messages(value: object) -> tuple[dict[str, str], ...]:
    if not isinstance(value, list) or len(value) < 2:
        raise ValueError("conversation must contain at least two messages")
    messages: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"role", "content"}:
            raise ValueError("message must contain exactly role and content")
        role = item["role"]
        content = item["content"]
        if role not in {"system", "user", "assistant"}:
            raise ValueError("unsupported conversation role")
        messages.append(
            {
                "role": cast(str, role),
                "content": _required_string(content, name="message content").replace(
                    "\r\n", "\n"
                ).replace("\r", "\n"),
            }
        )
    cursor = 0
    if messages[0]["role"] == "system":
        cursor = 1
    if cursor >= len(messages) or messages[cursor]["role"] != "user":
        raise ValueError("conversation must begin with an optional system then user")
    expected = "user"
    for message in messages[cursor:]:
        if message["role"] != expected:
            raise ValueError("conversation roles must alternate user and assistant")
        expected = "assistant" if expected == "user" else "user"
    if messages[-1]["role"] != "assistant":
        raise ValueError("conversation must end with assistant")
    return tuple(messages)


def _token_id(tokenizer: Tokenizer, token: str) -> int:
    value = tokenizer.token_to_id(token)
    if value is None:
        raise ValueError(f"tokenizer lacks {token}")
    return value


def _encode_messages(
    tokenizer: Tokenizer,
    messages: Sequence[Mapping[str, str]],
) -> TokenSequence:
    bos = _token_id(tokenizer, "<|bos|>")
    eos = _token_id(tokenizer, "<|eos|>")
    prompt = _token_id(tokenizer, "<|prompt|>")
    assistant = _token_id(tokenizer, "<|poem|>")
    ids = [bos]
    mask = [False]
    for message in messages:
        role = message["role"]
        content = message["content"]
        marker = assistant if role == "assistant" else prompt
        text = f"System:\n{content}" if role == "system" else content
        encoded = tokenizer.encode(text, add_special_tokens=False).ids
        ids.append(marker)
        mask.append(False)
        ids.extend(encoded)
        mask.extend(role == "assistant" for _ in encoded)
    ids.append(eos)
    mask.append(True)
    return TokenSequence(
        example_id="placeholder",
        boundary_key="sft",
        input_ids=tuple(ids),
        loss_mask=tuple(mask),
    )


def _normalised_text(text: str) -> str:
    return " ".join(text.casefold().split())


def _conversation_hash(messages: Sequence[Mapping[str, str]]) -> str:
    return hashlib.sha256(
        _canonical_json(
            [{"role": message["role"], "content": _normalised_text(message["content"])}
             for message in messages]
        ).encode()
    ).hexdigest()


def _assistant_messages(messages: Sequence[Mapping[str, str]]) -> tuple[str, ...]:
    return tuple(message["content"] for message in messages if message["role"] == "assistant")


def _smoltalk_example_id(source: SftSource, artifact: SourceArtifact, row_index: int) -> str:
    return (
        f"smol-smoltalk:{source.revision}:{artifact.split}:{Path(artifact.path).name}:"
        f"{row_index:09d}"
    )


def _iter_parquet_rows(path: Path) -> Iterable[tuple[int, dict[str, object]]]:
    row_index = 0
    parquet = pq.ParquetFile(path)
    if parquet.schema_arrow.names != ["messages", "source"]:
        raise ValueError(f"unexpected SmolTalk schema in {path}")
    for batch in parquet.iter_batches(batch_size=1024):
        for value in batch.to_pylist():
            if not isinstance(value, dict):
                raise TypeError(f"invalid SmolTalk row in {path}")
            yield row_index, cast(dict[str, object], value)
            row_index += 1


def _protected_texts(
    test_artifacts: Sequence[Path],
    heldout_paths: Sequence[Path],
    evaluation_suite: Path,
) -> tuple[tuple[str, str], ...]:
    values: list[tuple[str, str]] = []
    for path in test_artifacts:
        for row_index, row in _iter_parquet_rows(path):
            try:
                messages = _messages(row.get("messages"))
            except ValueError:
                continue
            values.extend(
                (text, f"smoltalk-test:{path.name}:{row_index}")
                for text in (
                    message["content"] for message in messages if message["role"] != "system"
                )
            )
    for path in heldout_paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                row = json.loads(line)
                target = row.get("poem_target")
                if isinstance(target, str) and target.strip():
                    values.append((target, f"track1-heldout:{path.name}:{line_number}"))
    suite = _read_json(evaluation_suite)
    cases = suite.get("cases")
    if not isinstance(cases, list):
        raise ValueError("evaluation suite cases must be a list")
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("evaluation suite case must be an object")
        prompt = case.get("prompt")
        case_id = case.get("case_id")
        if isinstance(prompt, str) and isinstance(case_id, str):
            values.append((prompt, f"evaluation-prompt:{case_id}"))
    return tuple(values)


def _synthetic_records(
    receipt_path: Path,
    tokenizer: Tokenizer,
) -> tuple[list[dict[str, object]], list[TokenSequence], set[str], int, int]:
    receipt = _read_json(receipt_path)
    dataset_path = receipt_path.parent / _required_string(
        receipt.get("dataset_filename", "dataset.jsonl"),
        name="synthetic dataset filename",
    )
    if file_hash(dataset_path) != _required_string(
        receipt.get("dataset_sha256"), name="synthetic dataset sha256"
    ):
        raise ValueError("synthetic dataset hash does not match its receipt")
    records: list[dict[str, object]] = []
    sequences: list[TokenSequence] = []
    response_hashes: set[str] = set()
    formatted_tokens = 0
    supervised_tokens = 0
    with dataset_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise TypeError("synthetic dataset row must be an object")
            example_id = _required_string(record.get("example_id"), name="synthetic example_id")
            messages = _messages(record.get("messages"))
            sequence = _encode_messages(tokenizer, messages)
            sequence = TokenSequence(
                example_id=example_id,
                boundary_key="sft",
                input_ids=sequence.input_ids,
                loss_mask=sequence.loss_mask,
            )
            counts = record.get("token_counts")
            if not isinstance(counts, dict) or counts.get("formatted") != len(sequence.input_ids):
                raise ValueError(f"synthetic token count mismatch for {example_id}")
            if counts.get("supervised") != sum(sequence.loss_mask):
                raise ValueError(f"synthetic supervised count mismatch for {example_id}")
            records.append(cast(dict[str, object], record))
            sequences.append(sequence)
            formatted_tokens += len(sequence.input_ids)
            supervised_tokens += sum(sequence.loss_mask)
            for response in _assistant_messages(messages):
                response_hashes.add(hashlib.sha256(_normalised_text(response).encode()).hexdigest())
    return records, sequences, response_hashes, formatted_tokens, supervised_tokens


def build_sft_mixture(
    *,
    source_config_path: Path,
    acquisition_directory: Path,
    synthetic_receipt_path: Path,
    tokenizer_path: Path,
    heldout_paths: Sequence[Path],
    evaluation_suite_path: Path,
    output_directory: Path,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    expected_tokenizer_sha256: str = TRACK1_SFT_TOKENIZER_SHA256,
) -> Path:
    """Build one deduplicated, protected, packed synthetic-plus-SmolTalk SFT artifact."""
    if target_tokens < 1:
        raise ValueError("target_tokens must be positive")
    if file_hash(tokenizer_path) != expected_tokenizer_sha256:
        raise ValueError("SFT mixture requires the frozen Track 1 tokenizer")
    verify_smoltalk_acquisition(source_config_path, acquisition_directory)
    _, source = _load_source_config(source_config_path)
    tokenizer = load_tokenizer(tokenizer_path)
    synthetic, sequences, response_hashes, formatted, supervised = _synthetic_records(
        synthetic_receipt_path,
        tokenizer,
    )
    if formatted >= target_tokens:
        raise ValueError("synthetic SFT data already meets or exceeds the target")

    test_paths = [
        acquisition_directory / artifact.path
        for artifact in source.artifacts
        if artifact.split == "test"
    ]
    train_artifacts = tuple(artifact for artifact in source.artifacts if artifact.split == "train")
    protected = _protected_texts(test_paths, heldout_paths, evaluation_suite_path)
    protected_index = LexicalFamilyIndex(protected)
    protected_conversations: set[str] = set()
    for path in test_paths:
        for _, row in _iter_parquet_rows(path):
            try:
                protected_conversations.add(_conversation_hash(_messages(row.get("messages"))))
            except ValueError:
                continue

    candidate_heap: list[tuple[int, str, str, int]] = []
    for artifact in train_artifacts:
        path = acquisition_directory / artifact.path
        for row_index, _ in _iter_parquet_rows(path):
            example_id = _smoltalk_example_id(source, artifact, row_index)
            sort_key = hashlib.sha256(f"{SELECTION_SEED}\0{example_id}".encode()).hexdigest()
            heap_entry = (-int(sort_key, 16), example_id, artifact.path, row_index)
            if len(candidate_heap) < CANDIDATE_POOL_SIZE:
                heapq.heappush(candidate_heap, heap_entry)
            elif heap_entry[0] > candidate_heap[0][0]:
                heapq.heapreplace(candidate_heap, heap_entry)
    candidate_rows = {
        (artifact_path, row_index)
        for _, _, artifact_path, row_index in candidate_heap
    }

    candidates: list[SmolCandidate] = []
    rejection_counts: Counter[str] = Counter()
    for artifact in train_artifacts:
        path = acquisition_directory / artifact.path
        for row_index, row in _iter_parquet_rows(path):
            if (artifact.path, row_index) not in candidate_rows:
                continue
            example_id = _smoltalk_example_id(source, artifact, row_index)
            try:
                messages = _messages(row.get("messages"))
            except (TypeError, ValueError):
                rejection_counts["invalid_conversation"] += 1
                continue
            conversation_hash = _conversation_hash(messages)
            if conversation_hash in protected_conversations:
                rejection_counts["protected_conversation"] += 1
                continue
            assistants = _assistant_messages(messages)
            assistant_hashes = tuple(
                hashlib.sha256(_normalised_text(text).encode()).hexdigest()
                for text in assistants
            )
            if len(assistant_hashes) != len(set(assistant_hashes)):
                rejection_counts["duplicate_assistant_within_conversation"] += 1
                continue
            protected_queries = tuple(
                message["content"] for message in messages if message["role"] != "system"
            )
            if any(protected_index.find_matches(text) for text in protected_queries):
                rejection_counts["protected_lexical_family"] += 1
                continue
            sequence = _encode_messages(tokenizer, messages)
            if len(sequence.input_ids) > SEQUENCE_LENGTH:
                rejection_counts["over_context"] += 1
                continue
            candidates.append(
                SmolCandidate(
                    sort_key=hashlib.sha256(f"{SELECTION_SEED}\0{example_id}".encode()).hexdigest(),
                    example_id=example_id,
                    artifact_path=artifact.path,
                    row_index=row_index,
                    formatted_tokens=len(sequence.input_ids),
                    supervised_tokens=sum(sequence.loss_mask),
                    conversation_hash=conversation_hash,
                    assistant_hashes=assistant_hashes,
                )
            )

    selected: dict[tuple[str, int], SmolCandidate] = {}
    seen_conversations: set[str] = set()
    for candidate in sorted(candidates, key=lambda value: (value.sort_key, value.example_id)):
        if candidate.conversation_hash in seen_conversations:
            rejection_counts["duplicate_conversation"] += 1
            continue
        if any(digest in response_hashes for digest in candidate.assistant_hashes):
            rejection_counts["duplicate_assistant"] += 1
            continue
        seen_conversations.add(candidate.conversation_hash)
        response_hashes.update(candidate.assistant_hashes)
        selected[(candidate.artifact_path, candidate.row_index)] = candidate
        formatted += candidate.formatted_tokens
        supervised += candidate.supervised_tokens
        if formatted >= target_tokens:
            break
    if formatted < target_tokens:
        raise ValueError(
            f"the {CANDIDATE_POOL_SIZE:,}-row deterministic SmolTalk candidate pool "
            f"reaches only {formatted:,} formatted tokens"
        )

    smoltalk_records: list[dict[str, object]] = []
    for artifact in train_artifacts:
        wanted = {
            row_index: candidate
            for (artifact_path, row_index), candidate in selected.items()
            if artifact_path == artifact.path
        }
        if not wanted:
            continue
        for row_index, row in _iter_parquet_rows(acquisition_directory / artifact.path):
            selected_candidate = wanted.get(row_index)
            if selected_candidate is None:
                continue
            messages = _messages(row.get("messages"))
            sequence = _encode_messages(tokenizer, messages)
            sequence = TokenSequence(
                example_id=selected_candidate.example_id,
                boundary_key="sft",
                input_ids=sequence.input_ids,
                loss_mask=sequence.loss_mask,
            )
            if (
                len(sequence.input_ids) != selected_candidate.formatted_tokens
                or sum(sequence.loss_mask) != selected_candidate.supervised_tokens
            ):
                raise ValueError(
                    f"SmolTalk token count drift for {selected_candidate.example_id}"
                )
            source_label = _required_string(row.get("source"), name="SmolTalk row source")
            smoltalk_records.append(
                {
                    "format_version": FORMAT_VERSION,
                    "example_id": selected_candidate.example_id,
                    "messages": list(messages),
                    "token_counts": {
                        "formatted": selected_candidate.formatted_tokens,
                        "supervised": selected_candidate.supervised_tokens,
                    },
                    "provenance": {
                        "kind": "external_sft",
                        "repository": source.repository,
                        "revision": source.revision,
                        "licence": source.licence,
                        "source": source_label,
                        "artifact": artifact.path,
                        "row_index": row_index,
                    },
                }
            )
            sequences.append(sequence)
    if len(smoltalk_records) != len(selected):
        raise ValueError("selected SmolTalk rows were not recovered from their Parquet files")

    output_directory.parent.mkdir(parents=True, exist_ok=True)
    if output_directory.exists():
        raise FileExistsError(f"SFT mixture output already exists: {output_directory}")
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_directory.name}.staging-",
            dir=output_directory.parent,
        )
    )
    try:
        dataset_path = staging / "dataset.jsonl"
        packs_path = staging / "packs.jsonl"
        all_records = [*synthetic, *smoltalk_records]
        _write_jsonl(dataset_path, all_records)
        packs = pack_sequences(sequences, sequence_length=SEQUENCE_LENGTH)
        write_packed_sequences(packs_path, packs)
        actual_formatted = sum(len(sequence.input_ids) for sequence in sequences)
        actual_supervised = sum(sum(sequence.loss_mask) for sequence in sequences)
        if actual_formatted != formatted or actual_supervised != supervised:
            raise ValueError("assembled SFT token totals drifted")
        _write_json(
            staging / "receipt.json",
            {
                "format_version": FORMAT_VERSION,
                "target_metric": "formatted",
                "target_tokens": target_tokens,
                "actual_formatted_tokens": actual_formatted,
                "target_overshoot": actual_formatted - target_tokens,
                "supervised_tokens": actual_supervised,
                "example_count": len(all_records),
                "synthetic_example_count": len(synthetic),
                "smoltalk_example_count": len(smoltalk_records),
                "pack_count": len(packs),
                "sequence_length": SEQUENCE_LENGTH,
                "tokenizer_sha256": file_hash(tokenizer_path),
                "selection_seed": SELECTION_SEED,
                "source_config_sha256": file_hash(source_config_path),
                "protection_inputs": [
                    {
                        "kind": "track1_heldout",
                        "path": str(path),
                        "sha256": file_hash(path),
                    }
                    for path in heldout_paths
                ]
                + [
                    {
                        "kind": "evaluation_suite",
                        "path": str(evaluation_suite_path),
                        "sha256": file_hash(evaluation_suite_path),
                    }
                ],
                "acquisition_receipt_sha256": file_hash(
                    acquisition_directory / "receipt.json"
                ),
                "synthetic_receipt_sha256": file_hash(synthetic_receipt_path),
                "rejection_counts": dict(sorted(rejection_counts.items())),
                "dataset_filename": dataset_path.name,
                "dataset_sha256": file_hash(dataset_path),
                "packs_filename": packs_path.name,
                "packs_sha256": file_hash(packs_path),
            },
        )
        staging.rename(output_directory)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return output_directory / "receipt.json"
