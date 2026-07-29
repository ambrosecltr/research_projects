"""Validated training access to a finalized SFT mixture."""

from __future__ import annotations

import json
import random
import sys
from array import array
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from tokenizers import Tokenizer

from poetry50m.config import file_hash
from poetry50m.training.stream import Batch, SkippedBatchStats

from .artifacts import read_packed_sequences
from .batch_stream import PreparedBatchStream
from .packing import PackedSequence
from .schema import ObjectiveMix


@dataclass(frozen=True, slots=True)
class SftTrainingArtifact:
    root: Path
    receipt: dict[str, Any]
    packs: tuple[PackedSequence, ...]

    @property
    def packs_path(self) -> Path:
        return self.root / cast(str, self.receipt["packs_filename"])

    @property
    def pack_count(self) -> int:
        return len(self.packs)

    @property
    def effective_input_tokens(self) -> int:
        return sum(len(pack.input_ids) - 1 for pack in self.packs)

    @property
    def data_sha256(self) -> str:
        return cast(str, self.receipt["packs_sha256"])


@dataclass(frozen=True, slots=True)
class BinarySftTrainingArtifact:
    root: Path
    receipt: dict[str, Any]

    @property
    def tokens_path(self) -> Path:
        return self.root / "train.tokens.bin"

    @property
    def losses_path(self) -> Path:
        return self.root / "train.loss.bin"

    @property
    def pack_count(self) -> int:
        return _required_integer(self.receipt, "row_count")

    @property
    def effective_input_tokens(self) -> int:
        return self.pack_count * (_required_integer(self.receipt, "row_width") - 1)

    @property
    def data_sha256(self) -> str:
        return cast(str, self.receipt["data_sha256"])


SftArtifact = SftTrainingArtifact | BinarySftTrainingArtifact


def _read_receipt(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError("SFT mixture receipt must be a JSON object")
    return cast(dict[str, Any], value)


def _required_integer(receipt: dict[str, Any], name: str) -> int:
    value = receipt.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"SFT mixture receipt has invalid {name}")
    return value


def _artifact_path(root: Path, receipt: dict[str, Any], name: str) -> Path:
    filename = receipt.get(name)
    if (
        not isinstance(filename, str)
        or not filename
        or Path(filename).name != filename
    ):
        raise ValueError(f"SFT mixture receipt has invalid {name}")
    return root / filename


def load_sft_training_artifact(
    root: Path,
    *,
    tokenizer_path: Path,
) -> SftArtifact:
    """Validate the sealed mixture and return its already-packed training rows."""
    receipt = _read_receipt(root / "receipt.json")
    if (
        receipt.get("format_version") == 2
        and receipt.get("artifact_type") == "general_sft_binary"
    ):
        return _load_binary_sft_artifact(root, receipt=receipt, tokenizer_path=tokenizer_path)
    if receipt.get("format_version") != 1:
        raise ValueError("unsupported SFT mixture receipt")
    dataset_path = _artifact_path(root, receipt, "dataset_filename")
    packs_path = _artifact_path(root, receipt, "packs_filename")
    if file_hash(dataset_path) != receipt.get("dataset_sha256"):
        raise ValueError("SFT dataset hash does not match its receipt")
    if file_hash(packs_path) != receipt.get("packs_sha256"):
        raise ValueError("SFT packs hash does not match their receipt")
    if file_hash(tokenizer_path) != receipt.get("tokenizer_sha256"):
        raise ValueError("SFT tokenizer hash does not match the mixture receipt")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    sequence_length = _required_integer(receipt, "sequence_length")
    packs = read_packed_sequences(packs_path)
    if len(packs) != _required_integer(receipt, "pack_count"):
        raise ValueError("SFT pack count does not match its receipt")
    if {pack.pack_id for pack in packs} != set(range(len(packs))):
        raise ValueError("SFT pack IDs must be unique and contiguous")
    if any(
        pack.boundary_key != "sft"
        or pack.objective != "conditional_poetry"
        or len(pack.input_ids) > sequence_length
        or max(pack.input_ids) >= tokenizer.get_vocab_size()
        for pack in packs
    ):
        raise ValueError("SFT packs violate the training contract")
    formatted_tokens = sum(len(pack.input_ids) for pack in packs)
    supervised_tokens = sum(sum(pack.loss_mask) for pack in packs)
    if formatted_tokens != _required_integer(receipt, "actual_formatted_tokens"):
        raise ValueError("SFT formatted-token total does not match its receipt")
    if supervised_tokens != _required_integer(receipt, "supervised_tokens"):
        raise ValueError("SFT supervised-token total does not match its receipt")
    return SftTrainingArtifact(root=root, receipt=receipt, packs=packs)


def _load_binary_sft_artifact(
    root: Path,
    *,
    receipt: dict[str, Any],
    tokenizer_path: Path,
) -> BinarySftTrainingArtifact:
    row_width = _required_integer(receipt, "row_width")
    row_count = _required_integer(receipt, "row_count")
    tokens_path = root / "train.tokens.bin"
    losses_path = root / "train.loss.bin"
    if (
        file_hash(tokenizer_path) != receipt.get("tokenizer_sha256")
        or file_hash(tokens_path) != receipt.get("tokens_sha256")
        or file_hash(losses_path) != receipt.get("losses_sha256")
    ):
        raise ValueError("binary SFT artifact hash mismatch")
    if (
        tokens_path.stat().st_size != row_count * row_width * 2
        or losses_path.stat().st_size != row_count * row_width
    ):
        raise ValueError("binary SFT artifact byte size mismatch")
    expected_data_hash = sha256(
        (
            cast(str, receipt["tokens_sha256"])
            + "\0"
            + cast(str, receipt["losses_sha256"])
        ).encode()
    ).hexdigest()
    if receipt.get("data_sha256") != expected_data_hash:
        raise ValueError("binary SFT data identity mismatch")
    return BinarySftTrainingArtifact(root=root, receipt=receipt)


def sft_batch_stream(
    artifact: SftArtifact,
    *,
    batch_size: int,
    pad_token_id: int,
    seed: int,
) -> PreparedBatchStream | BinarySftBatchStream:
    if isinstance(artifact, BinarySftTrainingArtifact):
        return BinarySftBatchStream(artifact, batch_size=batch_size, seed=seed)
    return PreparedBatchStream.from_artifact(
        str(artifact.packs_path),
        batch_size=batch_size,
        pad_token_id=pad_token_id,
        objective_mix=ObjectiveMix(
            conditional_poetry=1.0,
            auxiliary_prose_ntp=0.0,
            poetry_ntp=0.0,
        ),
        seed=seed,
    )


class BinarySftBatchStream:
    """Checkpointable shuffled access to response-only binary SFT rows."""

    def __init__(
        self,
        artifact: BinarySftTrainingArtifact,
        *,
        batch_size: int,
        seed: int,
    ) -> None:
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1
            or isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed < 0
        ):
            raise ValueError("binary SFT batch_size and seed are invalid")
        self._artifact = artifact
        self._batch_size = batch_size
        self._seed = seed
        self._row_count = artifact.pack_count
        self._row_width = _required_integer(artifact.receipt, "row_width")
        self._tokens = np.memmap(
            artifact.tokens_path,
            mode="r",
            dtype="<u2",
            shape=(self._row_count, self._row_width),
        )
        self._losses = np.memmap(
            artifact.losses_path,
            mode="r",
            dtype=np.uint8,
            shape=(self._row_count, self._row_width),
        )
        self._epoch = 0
        self._position = 0
        self._data_tokens_seen = 0
        self._order = self._order_for_epoch(0)
        self._stream_hash = sha256(
            json.dumps(
                {
                    "data_sha256": artifact.data_sha256,
                    "batch_size": batch_size,
                    "seed": seed,
                    "shuffle": "python-random-v1",
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()

    def _order_for_epoch(self, epoch: int) -> tuple[int, ...]:
        indices = list(range(self._row_count))
        random.Random(f"{self._seed}:sft:{epoch}").shuffle(indices)
        return tuple(indices)

    @staticmethod
    def _order_hash(order: tuple[int, ...]) -> str:
        values = array("Q", order)
        if sys.byteorder != "little":
            values.byteswap()
        return sha256(values.tobytes()).hexdigest()

    @property
    def order_digest(self) -> str:
        return sha256(
            json.dumps(
                {
                    "epoch": self._epoch,
                    "position": self._position,
                    "order_sha256": self._order_hash(self._order),
                    "data_tokens_seen": self._data_tokens_seen,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()

    def __iter__(self) -> BinarySftBatchStream:
        return self

    def _take_indices(self) -> tuple[int, ...]:
        remaining = self._row_count - self._position
        if remaining < self._batch_size:
            raise StopIteration(
                "binary SFT stream reached the end of its epoch before a complete batch"
            )
        end = self._position + self._batch_size
        selected = self._order[self._position : end]
        self._position = end
        if self._position == self._row_count:
            self._epoch += 1
            self._position = 0
            self._order = self._order_for_epoch(self._epoch)
        return selected

    def __next__(self) -> Batch:
        indices = self._take_indices()
        tokens = torch.from_numpy(
            np.asarray(self._tokens[list(indices)], dtype=np.int64)
        )
        losses = torch.from_numpy(
            np.asarray(self._losses[list(indices)], dtype=np.bool_)
        )
        input_ids = tokens[:, :-1]
        data_token_count = int(input_ids.numel())
        self._data_tokens_seen += data_token_count
        return {
            "input_ids": input_ids,
            "targets": tokens[:, 1:],
            "loss_mask": losses[:, 1:],
            "example_ids": indices,
            "data_token_count": data_token_count,
        }

    def skip_batches(self, count: int) -> SkippedBatchStats:
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("skip count must be a non-negative integer")
        data_tokens = 0
        for _ in range(count):
            indices = self._take_indices()
            batch_tokens = len(indices) * (self._row_width - 1)
            self._data_tokens_seen += batch_tokens
            data_tokens += batch_tokens
        return SkippedBatchStats(batch_count=count, data_token_count=data_tokens)

    def state_dict(self) -> dict[str, Any]:
        return {
            "format_version": 1,
            "data_sha256": self._artifact.data_sha256,
            "stream_hash": self._stream_hash,
            "epoch": self._epoch,
            "position": self._position,
            "data_tokens_seen": self._data_tokens_seen,
            "order_digest": self.order_digest,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        epoch = state.get("epoch")
        position = state.get("position")
        data_tokens_seen = state.get("data_tokens_seen")
        if (
            state.get("format_version") != 1
            or state.get("data_sha256") != self._artifact.data_sha256
            or state.get("stream_hash") != self._stream_hash
            or isinstance(epoch, bool)
            or not isinstance(epoch, int)
            or epoch < 0
            or isinstance(position, bool)
            or not isinstance(position, int)
            or not 0 <= position < self._row_count
            or isinstance(data_tokens_seen, bool)
            or not isinstance(data_tokens_seen, int)
            or data_tokens_seen < 0
        ):
            raise ValueError("invalid binary SFT stream state")
        self._epoch = epoch
        self._position = position
        self._data_tokens_seen = data_tokens_seen
        self._order = self._order_for_epoch(epoch)
        if state.get("order_digest") != self.order_digest:
            raise ValueError("binary SFT stream order mismatch")
