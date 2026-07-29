"""Streaming preparation of response-only Fineweb-Instruct SFT rows."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unicodedata
import urllib.request
from array import array
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import BinaryIO, cast

from huggingface_hub import HfApi, hf_hub_url
from huggingface_hub.hf_api import RepoFile
from tokenizers import Tokenizer

from poetry50m.config import file_hash, load_mapping


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n")).strip()


@dataclass(frozen=True, slots=True)
class GeneralSftConfig:
    repository: str
    revision: str
    filename: str
    size_bytes: int
    sha256: str
    licence: str
    selection_seed: str
    target_supervised_tokens: int
    context_length: int
    batch_size: int
    train_seed: int
    data_seed: int

    @classmethod
    def load(cls, path: Path) -> GeneralSftConfig:
        value = load_mapping(path)
        if set(value) != {"format_version", "source", "selection", "packing", "training"}:
            raise ValueError("general SFT config has unexpected keys")
        if value["format_version"] != 1:
            raise ValueError("general SFT format_version must be 1")
        source = value["source"]
        selection = value["selection"]
        packing = value["packing"]
        training = value["training"]
        for name, mapping, keys in (
            (
                "source",
                source,
                {"repository", "revision", "filename", "size_bytes", "sha256", "licence"},
            ),
            (
                "selection",
                selection,
                {"seed", "target_supervised_tokens"},
            ),
            ("packing", packing, {"context_length", "batch_size"}),
            ("training", training, {"seed", "data_seed"}),
        ):
            if not isinstance(mapping, dict) or set(mapping) != keys:
                raise ValueError(f"general SFT {name} has unexpected keys")
        source = cast(dict[str, object], source)
        selection = cast(dict[str, object], selection)
        packing = cast(dict[str, object], packing)
        training = cast(dict[str, object], training)

        def text(mapping: dict[str, object], name: str) -> str:
            item = mapping[name]
            if not isinstance(item, str) or not item:
                raise TypeError(f"{name} must be a non-empty string")
            return item

        def integer(mapping: dict[str, object], name: str) -> int:
            item = mapping[name]
            if isinstance(item, bool) or not isinstance(item, int) or item < 1:
                raise TypeError(f"{name} must be a positive integer")
            return item

        digest = text(source, "sha256")
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("source.sha256 must be lowercase SHA-256")
        return cls(
            repository=text(source, "repository"),
            revision=text(source, "revision"),
            filename=text(source, "filename"),
            size_bytes=integer(source, "size_bytes"),
            sha256=digest,
            licence=text(source, "licence"),
            selection_seed=text(selection, "seed"),
            target_supervised_tokens=integer(selection, "target_supervised_tokens"),
            context_length=integer(packing, "context_length"),
            batch_size=integer(packing, "batch_size"),
            train_seed=integer(training, "seed"),
            data_seed=integer(training, "data_seed"),
        )


@dataclass(slots=True)
class _SftWriter:
    tokens: BinaryIO
    losses: BinaryIO
    row_width: int
    batch_size: int
    token_buffer: array[int]
    loss_buffer: array[int]
    row_count: int = 0
    supervised_tokens: int = 0

    def append(self, token_ids: list[int], loss_mask: list[int]) -> None:
        if len(token_ids) != len(loss_mask):
            raise ValueError("SFT token and loss lengths differ")
        self.token_buffer.extend(token_ids)
        self.loss_buffer.extend(loss_mask)
        while len(self.token_buffer) >= self.row_width:
            token_row = self.token_buffer[: self.row_width]
            loss_row = self.loss_buffer[: self.row_width]
            del self.token_buffer[: self.row_width - 1]
            del self.loss_buffer[: self.row_width - 1]
            supervised = sum(loss_row[1:])
            if supervised == 0:
                continue
            if sys.byteorder != "little":
                token_row.byteswap()
            token_row.tofile(self.tokens)
            self.losses.write(bytes(loss_row))
            self.row_count += 1
            self.supervised_tokens += supervised

    def target_complete(self, target: int) -> bool:
        return self.supervised_tokens >= target and self.row_count % self.batch_size == 0


def _source_info(config: GeneralSftConfig) -> None:
    api = HfApi()
    info = api.dataset_info(config.repository, revision=config.revision)
    if info.sha != config.revision:
        raise ValueError("Fineweb-Instruct revision did not resolve to its pinned commit")
    paths = api.get_paths_info(
        config.repository,
        [config.filename],
        repo_type="dataset",
        revision=config.revision,
        expand=True,
    )
    if len(paths) != 1:
        raise ValueError("Fineweb-Instruct source file did not resolve")
    source = paths[0]
    if not isinstance(source, RepoFile):
        raise ValueError("Fineweb-Instruct source path is not a file")
    lfs = source.lfs
    if (
        source.size != config.size_bytes
        or lfs is None
        or lfs.size != config.size_bytes
        or lfs.sha256 != config.sha256
    ):
        raise ValueError("Fineweb-Instruct source identity changed")


def _source_lines(config: GeneralSftConfig) -> tuple[int, BinaryIO]:
    start = int.from_bytes(
        sha256(config.selection_seed.encode()).digest()[:8], "big"
    ) % config.size_bytes
    url = hf_hub_url(
        config.repository,
        config.filename,
        repo_type="dataset",
        revision=config.revision,
    )
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-"})
    response = urllib.request.urlopen(request, timeout=120)
    if start and getattr(response, "status", None) != 206:
        response.close()
        raise ValueError("Fineweb-Instruct source did not honor the byte-range request")
    if start:
        response.readline()
    return start, cast(BinaryIO, response)


def _training_config(config: GeneralSftConfig, *, one_epoch_steps: int) -> dict[str, object]:
    quarter = max(1, one_epoch_steps // 4)
    checkpoints = sorted(
        {
            quarter,
            max(1, one_epoch_steps // 2),
            max(1, 3 * one_epoch_steps // 4),
        }
    )
    return {
        "max_steps": one_epoch_steps,
        "learning_rate": 0.0001,
        "weight_decay": 0.0,
        "beta1": 0.9,
        "beta2": 0.95,
        "epsilon": 0.00000001,
        "warmup_steps": min(30, one_epoch_steps - 1),
        "min_learning_rate_ratio": 0.1,
        "gradient_accumulation_steps": 1,
        "max_grad_norm": 1.0,
        "device": "auto",
        "precision": "auto",
        "seed": config.train_seed,
        "deterministic": True,
        "log_every_steps": 1,
        "checkpoint_every_steps": 0,
        "trajectory_every_steps": 0,
        "checkpoint_steps": checkpoints,
        "trajectory_capture_steps": [],
        "analysis_every_steps": 0,
    }


def prepare_general_sft(
    *,
    config_path: Path,
    tokenizer_path: Path,
    output_directory: Path,
) -> Path:
    config = GeneralSftConfig.load(config_path)
    _source_info(config)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    required = {
        token: tokenizer.token_to_id(token)
        for token in ("<|bos|>", "<|eos|>", "<|user|>", "<|assistant|>")
    }
    if any(value is None for value in required.values()):
        raise ValueError("general tokenizer lacks SFT role tokens")
    ids = cast(dict[str, int], required)
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(f"general SFT output is not empty: {output_directory}")
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_directory.name}.preparing-",
            dir=output_directory.parent,
        )
    )
    tokens_path = staging / "train.tokens.bin"
    losses_path = staging / "train.loss.bin"
    counts: Counter[str] = Counter()
    seen_pairs: set[bytes] = set()
    formatted_source_tokens = 0
    byte_start: int | None = None
    try:
        with tokens_path.open("wb") as token_handle, losses_path.open("wb") as loss_handle:
            writer = _SftWriter(
                tokens=token_handle,
                losses=loss_handle,
                row_width=config.context_length + 1,
                batch_size=config.batch_size,
                token_buffer=array("H"),
                loss_buffer=array("B"),
            )
            byte_start, response = _source_lines(config)
            with response:
                for raw_line in response:
                    counts["rows_read"] += 1
                    try:
                        value: object = json.loads(raw_line)
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        counts["invalid_json"] += 1
                        continue
                    if not isinstance(value, dict):
                        counts["invalid_record"] += 1
                        continue
                    instruction = value.get("instruction")
                    answer = value.get("response")
                    if not isinstance(instruction, str) or not isinstance(answer, str):
                        counts["invalid_record"] += 1
                        continue
                    instruction = _normalize(instruction)
                    answer = _normalize(answer)
                    if not instruction or not answer:
                        counts["empty_text"] += 1
                        continue
                    if instruction.casefold() == answer.casefold():
                        counts["copied_answer"] += 1
                        continue
                    pair_hash = sha256(
                        instruction.encode() + b"\0" + answer.encode()
                    ).digest()
                    if pair_hash in seen_pairs:
                        counts["exact_duplicate"] += 1
                        continue
                    prefix = [
                        ids["<|bos|>"],
                        ids["<|user|>"],
                        *tokenizer.encode(instruction, add_special_tokens=False).ids,
                        ids["<|assistant|>"],
                    ]
                    if len(prefix) > config.context_length:
                        counts["instruction_over_context"] += 1
                        continue
                    target = [
                        *tokenizer.encode(answer, add_special_tokens=False).ids,
                        ids["<|eos|>"],
                    ]
                    if not target:
                        counts["empty_target"] += 1
                        continue
                    seen_pairs.add(pair_hash)
                    writer.append(
                        [*prefix, *target],
                        [0] * len(prefix) + [1] * len(target),
                    )
                    formatted_source_tokens += len(prefix) + len(target)
                    counts["accepted_examples"] += 1
                    if writer.target_complete(config.target_supervised_tokens):
                        break
            if not writer.target_complete(config.target_supervised_tokens):
                raise RuntimeError("Fineweb-Instruct ended before the SFT target was met")
            token_handle.flush()
            loss_handle.flush()
            os.fsync(token_handle.fileno())
            os.fsync(loss_handle.fileno())
        tokens_hash = file_hash(tokens_path)
        losses_hash = file_hash(losses_path)
        data_hash = sha256((tokens_hash + "\0" + losses_hash).encode()).hexdigest()
        one_epoch_steps = writer.row_count // config.batch_size
        receipt = {
            "format_version": 2,
            "artifact_type": "general_sft_binary",
            "repository": config.repository,
            "revision": config.revision,
            "licence": config.licence,
            "source_filename": config.filename,
            "source_size_bytes": config.size_bytes,
            "source_sha256": config.sha256,
            "source_start_byte": byte_start,
            "selection_seed": config.selection_seed,
            "target_metric": "supervised",
            "target_tokens": config.target_supervised_tokens,
            "actual_formatted_tokens": formatted_source_tokens,
            "supervised_tokens": writer.supervised_tokens,
            "effective_input_tokens": writer.row_count * config.context_length,
            "row_width": config.context_length + 1,
            "row_count": writer.row_count,
            "one_epoch_steps": one_epoch_steps,
            "tokenizer_sha256": file_hash(tokenizer_path),
            "config_sha256": file_hash(config_path),
            "tokens_sha256": tokens_hash,
            "losses_sha256": losses_hash,
            "data_sha256": data_hash,
            "counts": dict(sorted(counts.items())),
        }
        (staging / "receipt.json").write_text(
            json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        (staging / "one_epoch_train_config.json").write_text(
            json.dumps(
                _training_config(config, one_epoch_steps=one_epoch_steps),
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        from .sft_training import load_sft_training_artifact

        load_sft_training_artifact(staging, tokenizer_path=tokenizer_path)
        if output_directory.exists():
            output_directory.rmdir()
        os.replace(staging, output_directory)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_directory
