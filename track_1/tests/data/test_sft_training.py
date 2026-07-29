from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from poetry50m.config import file_hash
from poetry50m.data.artifacts import write_packed_sequences
from poetry50m.data.packing import PackedSequence
from poetry50m.data.sft_training import (
    BinarySftTrainingArtifact,
    load_sft_training_artifact,
    sft_batch_stream,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    tokenizer_path = tmp_path / "tokenizer.json"
    Tokenizer(
        WordLevel(
            {
                "<|pad|>": 0,
                "<|bos|>": 1,
                "<|eos|>": 2,
                "<|prompt|>": 3,
                "<|poem|>": 4,
                "bee": 5,
                "honey": 6,
            },
            unk_token="<|pad|>",
        )
    ).save(str(tokenizer_path))
    mixture = tmp_path / "mixture"
    mixture.mkdir()
    dataset = mixture / "dataset.jsonl"
    dataset.write_text('{"example_id":"example-1"}\n', encoding="utf-8")
    packs = mixture / "packs.jsonl"
    write_packed_sequences(
        packs,
        (
            PackedSequence(
                pack_id=0,
                boundary_key="sft",
                example_ids=("example-1",),
                input_ids=(1, 3, 5, 4, 6, 2),
                loss_mask=(False, False, False, False, True, True),
            ),
        ),
    )
    _write_json(
        mixture / "receipt.json",
        {
            "format_version": 1,
            "dataset_filename": dataset.name,
            "dataset_sha256": file_hash(dataset),
            "packs_filename": packs.name,
            "packs_sha256": file_hash(packs),
            "tokenizer_sha256": file_hash(tokenizer_path),
            "sequence_length": 8,
            "pack_count": 1,
            "actual_formatted_tokens": 6,
            "supervised_tokens": 2,
        },
    )
    return mixture, tokenizer_path


def test_sft_training_artifact_validates_and_streams_masks(tmp_path: Path) -> None:
    mixture, tokenizer_path = _fixture(tmp_path)
    artifact = load_sft_training_artifact(mixture, tokenizer_path=tokenizer_path)
    stream = sft_batch_stream(artifact, batch_size=1, pad_token_id=0, seed=11)

    batch = next(stream)

    assert batch["data_token_count"] == 5
    assert batch["loss_mask"].tolist() == [[False, False, False, True, True]]


def test_sft_training_artifact_rejects_tampered_packs(tmp_path: Path) -> None:
    mixture, tokenizer_path = _fixture(tmp_path)
    with (mixture / "packs.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("\n")

    with pytest.raises(ValueError, match="packs hash"):
        load_sft_training_artifact(mixture, tokenizer_path=tokenizer_path)


def test_binary_sft_artifact_streams_response_only_rows(tmp_path: Path) -> None:
    _, tokenizer_path = _fixture(tmp_path)
    root = tmp_path / "binary"
    root.mkdir()
    tokens = np.arange(18, dtype="<u2").reshape(2, 9)
    losses = np.zeros((2, 9), dtype=np.uint8)
    losses[:, -3:] = 1
    tokens.tofile(root / "train.tokens.bin")
    losses.tofile(root / "train.loss.bin")
    tokens_hash = file_hash(root / "train.tokens.bin")
    losses_hash = file_hash(root / "train.loss.bin")
    data_hash = sha256((tokens_hash + "\0" + losses_hash).encode()).hexdigest()
    _write_json(
        root / "receipt.json",
        {
            "format_version": 2,
            "artifact_type": "general_sft_binary",
            "row_width": 9,
            "row_count": 2,
            "actual_formatted_tokens": 18,
            "supervised_tokens": 6,
            "tokenizer_sha256": file_hash(tokenizer_path),
            "tokens_sha256": tokens_hash,
            "losses_sha256": losses_hash,
            "data_sha256": data_hash,
        },
    )

    artifact = load_sft_training_artifact(root, tokenizer_path=tokenizer_path)

    assert isinstance(artifact, BinarySftTrainingArtifact)
    stream = sft_batch_stream(artifact, batch_size=2, pad_token_id=0, seed=17)
    batch = next(stream)
    assert batch["input_ids"].shape == (2, 8)
    assert batch["loss_mask"].sum().item() == 6
