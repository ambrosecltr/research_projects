"""End-to-end deterministic preparation of a checkpointable training artifact."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

from .artifacts import (
    read_pairings,
    read_prompt_records,
    read_thought_records,
    write_conditional_examples,
    write_packed_sequences,
    write_prose_examples,
)
from .examples import build_auxiliary_prose_ntp_examples, build_conditional_examples
from .loaders import iter_manifest
from .packing import PackedSequence, pack_sequences
from .schema import ObjectiveMix, SplitName
from .splits import (
    LEXICAL_FAMILY_THRESHOLD,
    LexicalFamilyIndex,
    LexicalFamilyMatch,
    SplitRatios,
    split_examples,
    split_for_key,
)
from .tokenizer import (
    TokenizerSpec,
    encode_auxiliary_prose_ntp_example,
    encode_conditional_example,
    save_tokenizer,
    train_tokenizer,
)

_HeldoutLexicalPayload = tuple[SplitName, str, str, tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class PreparedDataConfig:
    split_ratios: SplitRatios
    split_salt: str
    tokenizer: TokenizerSpec
    sequence_length: int
    objective_mix: ObjectiveMix
    allow_synthetic: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.split_salt, str) or not self.split_salt:
            raise ValueError("split_salt must be a non-empty string")
        if (
            isinstance(self.sequence_length, bool)
            or not isinstance(self.sequence_length, int)
            or self.sequence_length < 2
        ):
            raise ValueError("sequence_length must be at least two")
        if not isinstance(self.allow_synthetic, bool):
            raise ValueError("allow_synthetic must be boolean")


@dataclass(frozen=True, slots=True)
class PreparedDataArtifact:
    root: Path
    metadata: dict[str, Any]

    @property
    def train_packs_path(self) -> Path:
        return self.root / "train.packed.jsonl"

    def conditional_path(self, split: SplitName) -> Path:
        return self.root / f"{split}.conditional.jsonl"

    def packed_path(self, split: SplitName) -> Path:
        return self.root / f"{split}.packed.jsonl"


def load_preparation_config(path: Path) -> PreparedDataConfig:
    raw_value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_value, dict) or not all(isinstance(key, str) for key in raw_value):
        raise ValueError("data config must be a JSON object")
    value = cast(dict[str, object], raw_value)
    allowed = {
        "format_version",
        "manifest_format",
        "manifest_schema",
        "split",
        "tokenizer",
        "packing",
        "objectives",
        "rights",
    }
    unknown = set(value).difference(allowed)
    if unknown:
        raise ValueError(f"unknown data config keys: {sorted(unknown)!r}")
    if (
        isinstance(value.get("format_version"), bool)
        or not isinstance(value.get("format_version"), int)
        or value.get("format_version") != 1
        or value.get("manifest_format") != "jsonl"
        or value.get("manifest_schema") != "SourceDocument"
    ):
        raise ValueError("unsupported prepared-data manifest format")

    def mapping_field(name: str, *, default_empty: bool = False) -> dict[str, object]:
        field = value.get(name, {} if default_empty else None)
        if not isinstance(field, dict) or not all(isinstance(key, str) for key in field):
            raise ValueError(f"data config {name} must be an object")
        return cast(dict[str, object], field)

    def number_field(mapping: Mapping[str, object], name: str) -> float:
        field = mapping.get(name)
        if isinstance(field, bool) or not isinstance(field, (int, float)):
            raise ValueError(f"{name} must be numeric")
        return float(field)

    def integer_field(mapping: Mapping[str, object], name: str) -> int:
        field = mapping.get(name)
        if isinstance(field, bool) or not isinstance(field, int):
            raise ValueError(f"{name} must be an integer")
        return field

    split = mapping_field("split")
    tokenizer = mapping_field("tokenizer")
    packing = mapping_field("packing")
    objectives = mapping_field("objectives")
    rights = mapping_field("rights", default_empty=True)
    nested_allowed = {
        "split": {"salt", "train", "validation", "test"},
        "tokenizer": {"vocab_size", "min_frequency", "special_tokens"},
        "packing": {"sequence_length"},
        "objectives": {"conditional_poetry", "auxiliary_prose_ntp"},
        "rights": {"allow_synthetic"},
    }
    for name, mapping in (
        ("split", split),
        ("tokenizer", tokenizer),
        ("packing", packing),
        ("objectives", objectives),
        ("rights", rights),
    ):
        unknown_nested = set(mapping).difference(nested_allowed[name])
        if unknown_nested:
            raise ValueError(f"unknown {name} config keys: {sorted(unknown_nested)!r}")
    salt = split.get("salt")
    special_tokens = tokenizer.get("special_tokens")
    allow_synthetic = rights.get("allow_synthetic", False)
    if not isinstance(salt, str):
        raise ValueError("split salt must be a string")
    if not isinstance(special_tokens, list) or not all(
        isinstance(item, str) for item in special_tokens
    ):
        raise ValueError("special_tokens must be a string list")
    if not isinstance(allow_synthetic, bool):
        raise ValueError("allow_synthetic must be boolean")
    return PreparedDataConfig(
        split_ratios=SplitRatios(
            number_field(split, "train"),
            number_field(split, "validation"),
            number_field(split, "test"),
        ),
        split_salt=salt,
        tokenizer=TokenizerSpec(
            vocab_size=integer_field(tokenizer, "vocab_size"),
            min_frequency=integer_field(tokenizer, "min_frequency"),
            special_tokens=tuple(special_tokens),
        ),
        sequence_length=integer_field(packing, "sequence_length"),
        objective_mix=ObjectiveMix(
            conditional_poetry=number_field(objectives, "conditional_poetry"),
            auxiliary_prose_ntp=number_field(objectives, "auxiliary_prose_ntp"),
        ),
        allow_synthetic=allow_synthetic,
    )


def _file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _match_sort_key(
    item: tuple[LexicalFamilyMatch, SplitName, str, str, tuple[str, ...]],
) -> tuple[float, str, str, str, str]:
    match, split_name, example_id, field_name, _ = item
    return (-match.score, match.metric, split_name, example_id, field_name)


def _train_objective_stats(
    packs: tuple[PackedSequence, ...],
) -> dict[str, dict[str, int | float]]:
    objective_names = ("conditional_poetry", "auxiliary_prose_ntp")
    supervised_by_objective = {
        name: sum(sum(pack.loss_mask) for pack in packs if pack.objective == name)
        for name in objective_names
    }
    total_supervised = sum(supervised_by_objective.values())
    if packs and total_supervised < 1:
        raise ValueError("prepared train packs contain no supervised tokens")
    return {
        name: {
            "pack_count": sum(pack.objective == name for pack in packs),
            "supervised_token_count": supervised_by_objective[name],
            "supervised_token_ratio": (
                supervised_by_objective[name] / total_supervised if total_supervised else 0.0
            ),
        }
        for name in objective_names
    }


def _prepare_data_in_directory(
    *,
    corpus_manifest: Path,
    prompt_records: Path,
    thought_records: Path,
    pairings: Path | None,
    output_directory: Path,
    config: PreparedDataConfig,
) -> PreparedDataArtifact:
    """Materialize the only data artifact consumed by the training stream."""
    documents = tuple(iter_manifest(corpus_manifest, allow_synthetic=config.allow_synthetic))
    prompts = read_prompt_records(prompt_records)
    thoughts = read_thought_records(thought_records)
    pairing_records = read_pairings(pairings) if pairings is not None else ()
    examples = build_conditional_examples(
        documents, prompts=prompts, thoughts=thoughts, pairings=pairing_records
    )
    split = split_examples(examples, config.split_ratios, salt=config.split_salt)
    document_splits: dict[str, SplitName] = {}
    for split_name, split_examples_for_name in split.items():
        for example in split_examples_for_name:
            for document_id in example.source_document_ids:
                previous = document_splits.setdefault(document_id, split_name)
                if previous != split_name:
                    raise ValueError(f"source document {document_id!r} spans dataset splits")
    output_directory.mkdir(parents=True, exist_ok=True)
    for name, values in split.items():
        write_conditional_examples(output_directory / f"{name}.conditional.jsonl", values)
    heldout_splits: tuple[SplitName, ...] = ("validation", "test")
    heldout_lexical_index: LexicalFamilyIndex[_HeldoutLexicalPayload] = LexicalFamilyIndex(
        (
            text,
            (
                split_name,
                example.example_id,
                field_name,
                tuple(sorted(example.source_document_ids)),
            ),
        )
        for split_name in heldout_splits
        for example in split[split_name]
        for field_name, text in (
            ("prompt", example.prompt),
            ("thought", example.thought or ""),
            ("poem_target", example.poem_target),
        )
        if text
    )
    held_out_texts = {
        text.casefold()
        for name in ("validation", "test")
        for example in split[name]
        for text in (example.prompt, example.thought or "", example.poem_target)
        if text
    }
    train_texts = [
        text
        for example in split["train"]
        for text in (example.prompt, example.thought or "", example.poem_target)
        if text and text.casefold() not in held_out_texts
    ]
    prose_examples_list = []
    excluded_prose: list[dict[str, object]] = []
    for prose_example in build_auxiliary_prose_ntp_examples(documents):
        owner = document_splits.get(
            prose_example.document_id,
            split_for_key(prose_example.document_id, config.split_ratios, salt=config.split_salt),
        )
        if owner != "train":
            heldout_examples = sorted(
                example.example_id
                for example in split[owner]
                if prose_example.document_id in example.source_document_ids
            )
            excluded_prose.append(
                {
                    "example_id": prose_example.example_id,
                    "block_id": prose_example.block_id,
                    "document_id": prose_example.document_id,
                    "reason": (
                        "heldout_document_family"
                        if heldout_examples
                        else "non_train_document_assignment"
                    ),
                    "evidence": {
                        "assigned_split": owner,
                        "heldout_example_ids": heldout_examples,
                    },
                }
            )
            continue
        matches: list[tuple[LexicalFamilyMatch, SplitName, str, str, tuple[str, ...]]] = []
        for hit in heldout_lexical_index.find_matches(prose_example.text):
            split_name, heldout_example_id, field_name, source_document_ids = hit.payload
            matches.append(
                (
                    hit.match,
                    split_name,
                    heldout_example_id,
                    field_name,
                    source_document_ids,
                )
            )
        if matches:
            match, split_name, heldout_example_id, field_name, source_document_ids = min(
                matches, key=_match_sort_key
            )
            excluded_prose.append(
                {
                    "example_id": prose_example.example_id,
                    "block_id": prose_example.block_id,
                    "document_id": prose_example.document_id,
                    "reason": "heldout_lexical_family",
                    "evidence": {
                        "metric": match.metric,
                        "score": match.score,
                        "threshold": (
                            1.0 if match.metric == "normalized_exact" else LEXICAL_FAMILY_THRESHOLD
                        ),
                        "shared_shingles": match.shared_shingles,
                        "comparison_shingles": match.comparison_shingles,
                        "heldout_split": split_name,
                        "heldout_example_id": heldout_example_id,
                        "heldout_field": field_name,
                        "heldout_source_document_ids": list(source_document_ids),
                    },
                }
            )
            continue
        prose_examples_list.append(prose_example)
    prose_examples = tuple(prose_examples_list)
    write_prose_examples(output_directory / "train.prose.jsonl", prose_examples)
    train_texts.extend(example.text for example in prose_examples)
    tokenizer = train_tokenizer(train_texts, config.tokenizer)
    actual_vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    if actual_vocab_size != config.tokenizer.vocab_size:
        raise ValueError("prepared tokenizer vocabulary is trainer-incompatible")
    tokenizer_path = output_directory / "tokenizer.json"
    save_tokenizer(tokenizer, tokenizer_path)
    prose_sequences = tuple(
        encode_auxiliary_prose_ntp_example(tokenizer, example) for example in prose_examples
    )
    packs_by_split: dict[str, tuple[PackedSequence, ...]] = {}
    for packed_split_name, values in split.items():
        sequences = tuple(encode_conditional_example(tokenizer, example) for example in values)
        packs_by_split[packed_split_name] = (
            pack_sequences(sequences, sequence_length=config.sequence_length) if sequences else ()
        )
    prose_packs = (
        pack_sequences(prose_sequences, sequence_length=config.sequence_length)
        if config.objective_mix.auxiliary_prose_ntp > 0
        else ()
    )
    offset_prose = tuple(
        PackedSequence(
            pack_id=len(packs_by_split["train"]) + index,
            boundary_key=pack.boundary_key,
            example_ids=pack.example_ids,
            input_ids=pack.input_ids,
            loss_mask=pack.loss_mask,
            objective=pack.objective,
        )
        for index, pack in enumerate(prose_packs)
    )
    packs_by_split["train"] = packs_by_split["train"] + offset_prose
    for artifact_split_name, packs_for_split in packs_by_split.items():
        write_packed_sequences(
            output_directory / f"{artifact_split_name}.packed.jsonl", packs_for_split
        )
    packs = packs_by_split["train"]
    metadata = {
        "format_version": 1,
        "config": asdict(config),
        "corpus_manifest_hash": _file_hash(corpus_manifest),
        "prompt_records_hash": _file_hash(prompt_records),
        "thought_records_hash": _file_hash(thought_records),
        "pairings_hash": _file_hash(pairings) if pairings is not None else None,
        "tokenizer_hash": _file_hash(tokenizer_path),
        "actual_vocab_size": actual_vocab_size,
        "split_counts": {name: len(values) for name, values in split.items()},
        "train_pack_count": len(packs),
        "train_objective_stats": _train_objective_stats(packs),
        "artifact_hashes": {
            path.name: _file_hash(path) for path in sorted(output_directory.glob("*.jsonl"))
        },
        "train_prose_block_ids": sorted(example.block_id for example in prose_examples),
        "excluded_prose": excluded_prose,
    }
    (output_directory / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return PreparedDataArtifact(output_directory, metadata)


def prepare_data(
    *,
    corpus_manifest: Path,
    prompt_records: Path,
    thought_records: Path,
    pairings: Path | None,
    output_directory: Path,
    config: PreparedDataConfig,
) -> PreparedDataArtifact:
    """Prepare in a clean sibling directory and publish with one atomic rename."""
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(f"prepared output is not empty: {output_directory}")
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_directory.name}.preparing-", dir=output_directory.parent)
    )
    expected = {
        "train.conditional.jsonl",
        "validation.conditional.jsonl",
        "test.conditional.jsonl",
        "train.packed.jsonl",
        "validation.packed.jsonl",
        "test.packed.jsonl",
        "tokenizer.json",
        "metadata.json",
        "train.prose.jsonl",
    }
    try:
        _prepare_data_in_directory(
            corpus_manifest=corpus_manifest,
            prompt_records=prompt_records,
            thought_records=thought_records,
            pairings=pairings,
            output_directory=temporary,
            config=config,
        )
        actual = {path.name for path in temporary.iterdir() if path.is_file()}
        if actual != expected:
            raise ValueError(f"prepared artifact file set mismatch: {sorted(actual ^ expected)!r}")
        load_prepared_data(temporary)
        if output_directory.exists():
            output_directory.rmdir()
        os.replace(temporary, output_directory)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return load_prepared_data(output_directory)


def load_prepared_data(path: Path) -> PreparedDataArtifact:
    metadata_path = path / "metadata.json"
    value = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("prepared metadata must be an object")
    hashes = value.get("artifact_hashes")
    if not isinstance(hashes, dict):
        raise ValueError("prepared metadata lacks artifact hashes")
    for name, expected in hashes.items():
        if (
            not isinstance(name, str)
            or not isinstance(expected, str)
            or _file_hash(path / name) != expected
        ):
            raise ValueError("prepared artifact hash does not match metadata")
    if value.get("tokenizer_hash") != _file_hash(path / "tokenizer.json"):
        raise ValueError("prepared tokenizer hash does not match metadata")
    return PreparedDataArtifact(path, value)
