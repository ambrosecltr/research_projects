from __future__ import annotations

import json
from pathlib import Path

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from poetry50m.config import file_hash
from poetry50m.data.artifacts import write_packed_sequences
from poetry50m.data.packing import PackedSequence
from poetry50m.data.sft_training import load_sft_training_artifact, sft_batch_stream


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
