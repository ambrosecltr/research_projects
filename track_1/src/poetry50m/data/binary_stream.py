"""Memory-mapped training batches for compact fixed-width token artifacts."""

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

from poetry50m.config import file_hash
from poetry50m.training.stream import Batch, SkippedBatchStats


@dataclass(frozen=True, slots=True)
class BinaryTokenArtifact:
    root: Path
    metadata: dict[str, Any]

    @property
    def tokenizer_path(self) -> Path:
        return self.root / "tokenizer.json"

    def tokens_path(self, split: str = "train") -> Path:
        return self.root / f"{split}.tokens.bin"

    def row_count(self, split: str = "train") -> int:
        splits = cast(dict[str, object], self.metadata["splits"])
        value = cast(dict[str, object], splits[split]).get("row_count")
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"binary artifact has invalid {split} row_count")
        return value


def load_binary_token_artifact(root: Path) -> BinaryTokenArtifact:
    metadata_path = root / "metadata.json"
    value: object = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError("binary artifact metadata must be a JSON object")
    metadata = cast(dict[str, Any], value)
    if (
        metadata.get("format_version") != 2
        or metadata.get("artifact_type") != "general_pretraining_binary"
        or metadata.get("token_dtype") != "uint16_le"
    ):
        raise ValueError("unsupported binary token artifact")
    row_width = metadata.get("row_width")
    if isinstance(row_width, bool) or not isinstance(row_width, int) or row_width < 2:
        raise ValueError("binary artifact has invalid row_width")
    hashes = metadata.get("artifact_hashes")
    splits = metadata.get("splits")
    if not isinstance(hashes, dict) or not isinstance(splits, dict):
        raise ValueError("binary artifact lacks hashes or splits")
    for name, expected in hashes.items():
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not isinstance(expected, str)
            or file_hash(root / name) != expected
        ):
            raise ValueError("binary artifact file hash mismatch")
    tokenizer_path = root / "tokenizer.json"
    if metadata.get("tokenizer_hash") != file_hash(tokenizer_path):
        raise ValueError("binary artifact tokenizer hash mismatch")
    for split, raw in splits.items():
        if not isinstance(split, str) or not isinstance(raw, dict):
            raise ValueError("binary artifact split metadata is invalid")
        row_count = raw.get("row_count")
        if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count < 1:
            raise ValueError(f"binary artifact has invalid {split} row_count")
        path = root / f"{split}.tokens.bin"
        if path.stat().st_size != row_count * row_width * 2:
            raise ValueError(f"binary artifact {split} byte size is invalid")
    return BinaryTokenArtifact(root=root, metadata=metadata)


class BinaryTokenBatchStream:
    """Shuffle fixed-width uint16 rows without loading their tokens into RAM."""

    def __init__(
        self,
        artifact: BinaryTokenArtifact,
        *,
        batch_size: int,
        seed: int,
        split: str = "train",
    ) -> None:
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1
            or isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed < 0
        ):
            raise ValueError("batch_size and seed are invalid")
        self._artifact = artifact
        self._split = split
        self._batch_size = batch_size
        self._seed = seed
        self._row_width = cast(int, artifact.metadata["row_width"])
        self._row_count = artifact.row_count(split)
        self._tokens = np.memmap(
            artifact.tokens_path(split),
            mode="r",
            dtype="<u2",
            shape=(self._row_count, self._row_width),
        )
        self._epoch = 0
        self._position = 0
        self._data_tokens_seen = 0
        self._order = self._order_for_epoch(0)
        self._artifact_hash = cast(
            dict[str, str], artifact.metadata["artifact_hashes"]
        )[f"{split}.tokens.bin"]
        self._stream_hash = sha256(
            json.dumps(
                {
                    "artifact_hash": self._artifact_hash,
                    "batch_size": batch_size,
                    "seed": seed,
                    "split": split,
                    "shuffle": "python-random-v1",
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()

    def _order_for_epoch(self, epoch: int) -> tuple[int, ...]:
        indices = list(range(self._row_count))
        random.Random(f"{self._seed}:{self._split}:{epoch}").shuffle(indices)
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

    @property
    def data_tokens_by_objective(self) -> dict[str, int]:
        return {"general_ntp": self._data_tokens_seen}

    def __iter__(self) -> BinaryTokenBatchStream:
        return self

    def _take_indices(self) -> tuple[int, ...]:
        remaining = self._row_count - self._position
        if remaining < self._batch_size:
            raise StopIteration(
                "binary stream reached the end of its epoch before a complete batch"
            )
        end = self._position + self._batch_size
        selected = self._order[self._position : end]
        self._position = end
        if self._position == self._row_count:
            self._epoch += 1
            self._position = 0
            self._order = self._order_for_epoch(self._epoch)
        return selected

    def _batch(self, indices: tuple[int, ...]) -> Batch:
        rows = np.asarray(self._tokens[list(indices)], dtype=np.int64)
        tensor = torch.from_numpy(rows)
        input_ids = tensor[:, :-1]
        targets = tensor[:, 1:]
        data_token_count = int(input_ids.numel())
        return {
            "input_ids": input_ids,
            "targets": targets,
            "loss_mask": torch.ones_like(input_ids, dtype=torch.bool),
            "example_ids": indices,
            "data_token_count": data_token_count,
        }

    def __next__(self) -> Batch:
        indices = self._take_indices()
        batch = self._batch(indices)
        self._data_tokens_seen += batch["data_token_count"]
        return batch

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
            "artifact_hash": self._artifact_hash,
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
            or state.get("artifact_hash") != self._artifact_hash
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
            raise ValueError("invalid binary-stream state")
        self._epoch = epoch
        self._position = position
        self._data_tokens_seen = data_tokens_seen
        self._order = self._order_for_epoch(epoch)
        if state.get("order_digest") != self.order_digest:
            raise ValueError("binary-stream order mismatch")
