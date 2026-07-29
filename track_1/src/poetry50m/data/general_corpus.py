"""Streaming preparation of a compact Ultra-FineWeb general pretraining corpus."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unicodedata
from array import array
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import BinaryIO, cast

import pyarrow.parquet as pq  # type: ignore[import-untyped]
from huggingface_hub import HfApi, hf_hub_download
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

from poetry50m.config import file_hash, load_mapping

GENERAL_SPECIAL_TOKENS = (
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|mask|>",
)
_SOURCE_NAMES = ("multi_style", "qa")
_SPLIT_NAMES = ("train", "validation", "test")


def _exact_mapping(value: object, *, name: str, keys: set[str]) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a JSON object")
    mapping = cast(dict[str, object], value)
    if set(mapping) != keys:
        raise ValueError(f"{name} must contain exactly {sorted(keys)}")
    return mapping


def _integer(mapping: Mapping[str, object], name: str, *, minimum: int = 1) -> int:
    value = mapping[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _string(mapping: Mapping[str, object], name: str) -> str:
    value = mapping[name]
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _row_counts(value: object, *, name: str) -> dict[str, int]:
    mapping = _exact_mapping(value, name=name, keys=set(_SOURCE_NAMES))
    return {source: _integer(mapping, source) for source in _SOURCE_NAMES}


@dataclass(frozen=True, slots=True)
class GeneralCorpusConfig:
    repository: str
    revision: str
    source_prefixes: dict[str, str]
    expected_styles: dict[str, str]
    shard_order_seed: str
    split_salt: str
    split_ratios: dict[str, float]
    vocab_size: int
    min_frequency: int
    special_tokens: tuple[str, ...]
    tokenizer_sample_utf8_bytes: int
    context_length: int
    batch_size: int
    train_rows: dict[str, int]
    heldout_rows: dict[str, dict[str, int]]

    @property
    def row_width(self) -> int:
        return self.context_length + 1

    @classmethod
    def load(cls, path: Path) -> GeneralCorpusConfig:
        value = load_mapping(path)
        root = _exact_mapping(
            value,
            name="general corpus config",
            keys={
                "format_version",
                "source",
                "selection",
                "split",
                "tokenizer",
                "packing",
                "rows",
            },
        )
        if root["format_version"] != 1:
            raise ValueError("general corpus format_version must be 1")
        source = _exact_mapping(
            root["source"],
            name="source",
            keys={"repository", "revision", "prefixes", "styles"},
        )
        prefixes = _exact_mapping(
            source["prefixes"], name="source.prefixes", keys=set(_SOURCE_NAMES)
        )
        styles = _exact_mapping(source["styles"], name="source.styles", keys=set(_SOURCE_NAMES))
        selection = _exact_mapping(
            root["selection"],
            name="selection",
            keys={"shard_order_seed"},
        )
        split = _exact_mapping(
            root["split"],
            name="split",
            keys={"salt", "train", "validation", "test"},
        )
        tokenizer = _exact_mapping(
            root["tokenizer"],
            name="tokenizer",
            keys={"vocab_size", "min_frequency", "special_tokens", "sample_utf8_bytes"},
        )
        packing = _exact_mapping(
            root["packing"], name="packing", keys={"context_length", "batch_size"}
        )
        rows = _exact_mapping(
            root["rows"], name="rows", keys={"train", "validation", "test"}
        )
        raw_specials = tokenizer["special_tokens"]
        if not isinstance(raw_specials, list) or any(
            not isinstance(token, str) for token in raw_specials
        ):
            raise TypeError("tokenizer.special_tokens must be a string list")
        special_tokens = tuple(cast(list[str], raw_specials))
        if special_tokens != GENERAL_SPECIAL_TOKENS:
            raise ValueError("tokenizer.special_tokens do not match the general model contract")
        ratios: dict[str, float] = {}
        for name in _SPLIT_NAMES:
            ratio = split[name]
            if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or ratio <= 0:
                raise ValueError(f"split.{name} must be positive")
            ratios[name] = float(ratio)
        if abs(sum(ratios.values()) - 1.0) > 1e-9:
            raise ValueError("split ratios must sum to one")
        train_rows = _row_counts(rows["train"], name="rows.train")
        heldout_rows = {
            name: _row_counts(rows[name], name=f"rows.{name}")
            for name in ("validation", "test")
        }
        batch_size = _integer(packing, "batch_size")
        if sum(train_rows.values()) % batch_size:
            raise ValueError("total train rows must be divisible by packing.batch_size")
        return cls(
            repository=_string(source, "repository"),
            revision=_string(source, "revision"),
            source_prefixes={name: _string(prefixes, name) for name in _SOURCE_NAMES},
            expected_styles={name: _string(styles, name) for name in _SOURCE_NAMES},
            shard_order_seed=_string(selection, "shard_order_seed"),
            split_salt=_string(split, "salt"),
            split_ratios=ratios,
            vocab_size=_integer(tokenizer, "vocab_size"),
            min_frequency=_integer(tokenizer, "min_frequency"),
            special_tokens=special_tokens,
            tokenizer_sample_utf8_bytes=_integer(tokenizer, "sample_utf8_bytes"),
            context_length=_integer(packing, "context_length"),
            batch_size=batch_size,
            train_rows=train_rows,
            heldout_rows=heldout_rows,
        )


def _normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n")).strip()


def _priority(seed: str, value: str) -> bytes:
    return sha256(f"{seed}\0{value}".encode()).digest()


def _split_for_uid(uid: str, config: GeneralCorpusConfig) -> str:
    point = int.from_bytes(_priority(config.split_salt, uid)[:8], "big") / 2**64
    train_end = config.split_ratios["train"]
    validation_end = train_end + config.split_ratios["validation"]
    if point < train_end:
        return "train"
    if point < validation_end:
        return "validation"
    return "test"


def _train_tokenizer(
    sample_paths: list[Path],
    *,
    config: GeneralCorpusConfig,
    output: Path,
) -> None:
    tokenizer = Tokenizer(models.BPE(unk_token=None, byte_fallback=True))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    trainer_factory = cast(Callable[..., trainers.BpeTrainer], trainers.BpeTrainer)
    trainer = trainer_factory(
        vocab_size=config.vocab_size,
        min_frequency=config.min_frequency,
        special_tokens=list(config.special_tokens),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train([str(path) for path in sample_paths], trainer=trainer)
    actual = tokenizer.get_vocab_size(with_added_tokens=True)
    if actual < config.vocab_size:
        tokenizer.add_tokens(
            [f"<|reserved_{index:05d}|>" for index in range(config.vocab_size - actual)]
        )
    if tokenizer.get_vocab_size(with_added_tokens=True) != config.vocab_size:
        raise RuntimeError("general tokenizer vocabulary does not match its config")
    tokenizer.save(str(output), pretty=True)


def _validate_tokenizer(path: Path, *, config: GeneralCorpusConfig) -> Tokenizer:
    tokenizer = Tokenizer.from_file(str(path))
    if tokenizer.get_vocab_size(with_added_tokens=True) != config.vocab_size:
        raise ValueError("general tokenizer vocabulary does not match its config")
    for token in config.special_tokens:
        if tokenizer.token_to_id(token) is None:
            raise ValueError(f"general tokenizer lacks required token {token}")
    return tokenizer


def _parquet_rows(path: Path) -> Iterator[tuple[str, str, str]]:
    parquet = pq.ParquetFile(path)
    if parquet.schema_arrow.names != ["uid", "content", "style"]:
        raise ValueError(f"unexpected Ultra-FineWeb schema in {path.name}")
    for batch in parquet.iter_batches(
        batch_size=2_048,
        columns=["uid", "content", "style"],
    ):
        values = batch.to_pydict()
        for uid, content, style in zip(
            values["uid"], values["content"], values["style"], strict=True
        ):
            if (
                not isinstance(uid, str)
                or not isinstance(content, str)
                or not isinstance(style, str)
            ):
                raise TypeError(f"invalid Ultra-FineWeb row in {path.name}")
            yield uid, content, style


def _download_shard(
    *,
    config: GeneralCorpusConfig,
    filename: str,
    scratch: Path,
) -> tuple[Path, dict[str, object]]:
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True)
    path = Path(
        hf_hub_download(
            repo_id=config.repository,
            filename=filename,
            repo_type="dataset",
            revision=config.revision,
            local_dir=scratch,
        )
    )
    return path, {
        "path": filename,
        "size_bytes": path.stat().st_size,
        "sha256": file_hash(path),
    }


def _source_paths(config: GeneralCorpusConfig) -> dict[str, list[str]]:
    api = HfApi()
    info = api.dataset_info(config.repository, revision=config.revision)
    if info.sha != config.revision:
        raise ValueError("Ultra-FineWeb revision did not resolve to the pinned commit")
    files = api.list_repo_files(
        config.repository,
        repo_type="dataset",
        revision=config.revision,
    )
    result: dict[str, list[str]] = {}
    for source in _SOURCE_NAMES:
        prefix = config.source_prefixes[source]
        paths = [
            name for name in files if name.startswith(prefix) and name.endswith(".parquet")
        ]
        if not paths:
            raise ValueError(f"no pinned Ultra-FineWeb shards found for {source}")
        result[source] = sorted(
            paths,
            key=lambda name: (_priority(config.shard_order_seed, f"{source}:{name}"), name),
        )
    return result


def _write_tokenizer_samples(
    *,
    config: GeneralCorpusConfig,
    paths_by_source: dict[str, list[str]],
    scratch: Path,
) -> tuple[list[Path], list[dict[str, object]]]:
    total_train_rows = sum(config.train_rows.values())
    samples: list[Path] = []
    receipts: list[dict[str, object]] = []
    for source in _SOURCE_NAMES:
        target = round(
            config.tokenizer_sample_utf8_bytes
            * config.train_rows[source]
            / total_train_rows
        )
        sample_path = scratch.parent / f"tokenizer-sample-{source}.txt"
        written = 0
        shard_index = 0
        with sample_path.open("wb") as handle:
            while written < target:
                if shard_index >= len(paths_by_source[source]):
                    raise RuntimeError(f"not enough {source} data for tokenizer sample")
                shard_name = paths_by_source[source][shard_index]
                shard_path, receipt = _download_shard(
                    config=config,
                    filename=shard_name,
                    scratch=scratch,
                )
                rows = 0
                for uid, raw_content, style in _parquet_rows(shard_path):
                    del uid
                    if style != config.expected_styles[source]:
                        raise ValueError(f"unexpected {source} style {style!r}")
                    content = _normalize_text(raw_content)
                    if not content:
                        continue
                    encoded = (content + "\n").encode("utf-8")
                    remaining = target - written
                    handle.write(encoded[:remaining])
                    written += min(len(encoded), remaining)
                    rows += 1
                    if written >= target:
                        break
                receipt.update(
                    {
                        "purpose": "tokenizer_sample",
                        "source": source,
                        "rows_read": rows,
                    }
                )
                receipts.append(receipt)
                shard_index += 1
        samples.append(sample_path)
    return samples, receipts


@dataclass(slots=True)
class _SplitWriter:
    path: Path
    target_rows: int
    row_width: int
    handle: BinaryIO
    buffer: array[int]
    rows_written: int = 0
    documents_written: int = 0

    @classmethod
    def open(cls, path: Path, *, target_rows: int, row_width: int) -> _SplitWriter:
        return cls(
            path=path,
            target_rows=target_rows,
            row_width=row_width,
            handle=path.open("wb"),
            buffer=array("H"),
        )

    @property
    def complete(self) -> bool:
        return self.rows_written == self.target_rows

    def append(self, token_ids: list[int]) -> None:
        if self.complete:
            return
        self.buffer.extend(token_ids)
        self.documents_written += 1
        while len(self.buffer) >= self.row_width and not self.complete:
            row = self.buffer[: self.row_width]
            if row.itemsize != 2:
                raise RuntimeError("uint16 token writer has an unexpected item size")
            if sys.byteorder != "little":
                row.byteswap()
            row.tofile(self.handle)
            del self.buffer[: self.row_width - 1]
            self.rows_written += 1

    def close(self) -> None:
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()


def _target_rows(config: GeneralCorpusConfig, split: str, source: str) -> int:
    if split == "train":
        return config.train_rows[source]
    return config.heldout_rows[split][source]


def _encode_rows(
    tokenizer: Tokenizer,
    accepted: list[tuple[_SplitWriter, str]],
    *,
    bos_id: int,
    eos_id: int,
) -> None:
    if not accepted:
        return
    encodings = tokenizer.encode_batch(
        [content for _, content in accepted],
        add_special_tokens=False,
    )
    for (writer, _), encoding in zip(accepted, encodings, strict=True):
        writer.append([bos_id, *encoding.ids, eos_id])
    accepted.clear()


def _build_token_files(
    *,
    config: GeneralCorpusConfig,
    paths_by_source: dict[str, list[str]],
    tokenizer_path: Path,
    output: Path,
    scratch: Path,
) -> dict[str, object]:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    bos_id = tokenizer.token_to_id("<|bos|>")
    eos_id = tokenizer.token_to_id("<|eos|>")
    if bos_id is None or eos_id is None:
        raise ValueError("general tokenizer lacks BOS or EOS")
    writers = {
        (split, source): _SplitWriter.open(
            output / f".{split}.{source}.bin",
            target_rows=_target_rows(config, split, source),
            row_width=config.row_width,
        )
        for split in _SPLIT_NAMES
        for source in _SOURCE_NAMES
    }
    seen_uids: set[str] = set()
    seen_content_hashes: set[bytes] = set()
    duplicate_uids = 0
    duplicate_contents = 0
    empty_contents = 0
    shard_receipts: list[dict[str, object]] = []
    try:
        for source in _SOURCE_NAMES:
            for shard_name in paths_by_source[source]:
                source_writers = [writers[(split, source)] for split in _SPLIT_NAMES]
                if all(writer.complete for writer in source_writers):
                    break
                print(f"processing {source} shard {shard_name}", flush=True)
                shard_path, receipt = _download_shard(
                    config=config,
                    filename=shard_name,
                    scratch=scratch,
                )
                rows_read = 0
                accepted: list[tuple[_SplitWriter, str]] = []

                for uid, raw_content, style in _parquet_rows(shard_path):
                    rows_read += 1
                    if style != config.expected_styles[source]:
                        raise ValueError(f"unexpected {source} style {style!r}")
                    split = _split_for_uid(uid, config)
                    writer = writers[(split, source)]
                    if writer.complete:
                        continue
                    if uid in seen_uids:
                        duplicate_uids += 1
                        continue
                    content = _normalize_text(raw_content)
                    if not content:
                        empty_contents += 1
                        continue
                    content_hash = sha256(content.encode()).digest()
                    if content_hash in seen_content_hashes:
                        duplicate_contents += 1
                        continue
                    seen_uids.add(uid)
                    seen_content_hashes.add(content_hash)
                    accepted.append((writer, content))
                    if len(accepted) == 512:
                        _encode_rows(
                            tokenizer,
                            accepted,
                            bos_id=bos_id,
                            eos_id=eos_id,
                        )
                _encode_rows(tokenizer, accepted, bos_id=bos_id, eos_id=eos_id)
                receipt.update(
                    {
                        "purpose": "corpus",
                        "source": source,
                        "rows_read": rows_read,
                        "rows_after_shard": {
                            split: writers[(split, source)].rows_written
                            for split in _SPLIT_NAMES
                        },
                    }
                )
                shard_receipts.append(receipt)
            incomplete = {
                split: writer.rows_written
                for (split, writer_source), writer in writers.items()
                if writer_source == source and not writer.complete
            }
            if incomplete:
                raise RuntimeError(f"not enough {source} data to satisfy row targets: {incomplete}")
    finally:
        for writer in writers.values():
            writer.close()
    for split in _SPLIT_NAMES:
        combined = output / f"{split}.tokens.bin"
        with combined.open("wb") as destination:
            for source in _SOURCE_NAMES:
                temporary = writers[(split, source)].path
                with temporary.open("rb") as source_handle:
                    shutil.copyfileobj(source_handle, destination, length=1 << 20)
                temporary.unlink()
    return {
        "shards": shard_receipts,
        "duplicate_uids_removed": duplicate_uids,
        "duplicate_contents_removed": duplicate_contents,
        "empty_contents_removed": empty_contents,
        "unique_document_count": len(seen_content_hashes),
        "documents_by_split_and_source": {
            split: {
                source: writers[(split, source)].documents_written for source in _SOURCE_NAMES
            }
            for split in _SPLIT_NAMES
        },
    }


def prepare_general_corpus(
    *,
    config_path: Path,
    output_directory: Path,
    scratch_directory: Path,
    tokenizer_path: Path | None = None,
) -> Path:
    """Build and validate the complete binary artifact through one atomic publish."""
    config = GeneralCorpusConfig.load(config_path)
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(f"general corpus output is not empty: {output_directory}")
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_directory.name}.preparing-",
            dir=output_directory.parent,
        )
    )
    scratch = scratch_directory.resolve()
    if scratch == output_directory.resolve() or scratch == temporary.resolve():
        raise ValueError("scratch directory must be separate from output")
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True)
    try:
        paths_by_source = _source_paths(config)
        artifact_tokenizer_path = temporary / "tokenizer.json"
        if tokenizer_path is None:
            sample_paths, tokenizer_receipts = _write_tokenizer_samples(
                config=config,
                paths_by_source=paths_by_source,
                scratch=scratch / "download",
            )
            print("training the general 8M tokenizer", flush=True)
            _train_tokenizer(
                sample_paths,
                config=config,
                output=artifact_tokenizer_path,
            )
            tokenizer_provenance: dict[str, object] = {
                "mode": "trained",
                "sample_utf8_bytes": config.tokenizer_sample_utf8_bytes,
                "source_shards": tokenizer_receipts,
            }
        else:
            source_tokenizer_path = tokenizer_path.expanduser().resolve(strict=True)
            _validate_tokenizer(source_tokenizer_path, config=config)
            shutil.copyfile(source_tokenizer_path, artifact_tokenizer_path)
            tokenizer_provenance = {
                "mode": "reused",
                "source_sha256": file_hash(source_tokenizer_path),
            }
        _validate_tokenizer(artifact_tokenizer_path, config=config)
        build_receipt = _build_token_files(
            config=config,
            paths_by_source=paths_by_source,
            tokenizer_path=artifact_tokenizer_path,
            output=temporary,
            scratch=scratch / "download",
        )
        split_metadata = {
            split: {
                "row_count": sum(
                    _target_rows(config, split, source) for source in _SOURCE_NAMES
                ),
                "data_token_count": sum(
                    _target_rows(config, split, source) for source in _SOURCE_NAMES
                )
                * config.context_length,
                "source_rows": {
                    source: _target_rows(config, split, source) for source in _SOURCE_NAMES
                },
            }
            for split in _SPLIT_NAMES
        }
        artifact_names = [
            "tokenizer.json",
            *(f"{split}.tokens.bin" for split in _SPLIT_NAMES),
        ]
        metadata = {
            "format_version": 2,
            "artifact_type": "general_pretraining_binary",
            "token_dtype": "uint16_le",
            "row_width": config.row_width,
            "context_length": config.context_length,
            "batch_size": config.batch_size,
            "repository": config.repository,
            "revision": config.revision,
            "config": asdict(config),
            "config_sha256": file_hash(config_path),
            "tokenizer_hash": file_hash(artifact_tokenizer_path),
            "tokenizer_training": tokenizer_provenance,
            "splits": split_metadata,
            "train_objective_stats": {
                "general_ntp": {
                    "data_token_count": split_metadata["train"]["data_token_count"],
                    "data_token_ratio": 1.0,
                    "supervised_token_count": split_metadata["train"]["data_token_count"],
                    "supervised_token_ratio": 1.0,
                    "pack_count": split_metadata["train"]["row_count"],
                }
            },
            "source_build": build_receipt,
            "artifact_hashes": {
                name: file_hash(temporary / name) for name in artifact_names
            },
        }
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        from .binary_stream import load_binary_token_artifact

        load_binary_token_artifact(temporary)
        if output_directory.exists():
            output_directory.rmdir()
        os.replace(temporary, output_directory)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return output_directory
