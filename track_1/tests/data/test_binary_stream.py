from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer, models

from poetry50m.config import file_hash
from poetry50m.data.binary_stream import BinaryTokenBatchStream, load_binary_token_artifact


def _artifact(root: Path) -> Path:
    root.mkdir()
    tokenizer = Tokenizer(models.WordLevel({"<|pad|>": 0, "a": 1}, unk_token=None))
    tokenizer.save(str(root / "tokenizer.json"))
    rows = np.arange(45, dtype="<u2").reshape(5, 9)
    for split in ("train", "validation", "test"):
        rows.tofile(root / f"{split}.tokens.bin")
    artifact_hashes = {
        "tokenizer.json": file_hash(root / "tokenizer.json"),
        **{
            f"{split}.tokens.bin": file_hash(root / f"{split}.tokens.bin")
            for split in ("train", "validation", "test")
        },
    }
    metadata = {
        "format_version": 2,
        "artifact_type": "general_pretraining_binary",
        "token_dtype": "uint16_le",
        "row_width": 9,
        "tokenizer_hash": artifact_hashes["tokenizer.json"],
        "artifact_hashes": artifact_hashes,
        "splits": {
            split: {"row_count": 5} for split in ("train", "validation", "test")
        },
    }
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return root


def test_binary_stream_is_deterministic_and_checkpointable(tmp_path: Path) -> None:
    artifact = load_binary_token_artifact(_artifact(tmp_path / "prepared"))
    first = BinaryTokenBatchStream(artifact, batch_size=2, seed=41)
    second = BinaryTokenBatchStream(artifact, batch_size=2, seed=41)

    first_batch = next(first)
    assert first_batch["input_ids"].shape == (2, 8)
    assert first_batch["targets"].shape == (2, 8)
    assert first_batch["data_token_count"] == 16
    assert first_batch["example_ids"] == next(second)["example_ids"]

    state = first.state_dict()
    restored = BinaryTokenBatchStream(artifact, batch_size=2, seed=41)
    restored.load_state_dict(state)
    assert next(first)["example_ids"] == next(restored)["example_ids"]


def test_binary_stream_rejects_incomplete_final_batch(tmp_path: Path) -> None:
    artifact = load_binary_token_artifact(_artifact(tmp_path / "prepared"))
    stream = BinaryTokenBatchStream(artifact, batch_size=2, seed=9)
    next(stream)
    next(stream)
    with pytest.raises(StopIteration, match="complete batch"):
        next(stream)


def test_binary_artifact_detects_token_file_change(tmp_path: Path) -> None:
    root = _artifact(tmp_path / "prepared")
    with (root / "train.tokens.bin").open("ab") as handle:
        handle.write(b"\0\0")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_binary_token_artifact(root)


def test_fixture_hash_is_stable(tmp_path: Path) -> None:
    root = _artifact(tmp_path / "prepared")
    digest = sha256((root / "train.tokens.bin").read_bytes()).hexdigest()
    assert digest == file_hash(root / "train.tokens.bin")
