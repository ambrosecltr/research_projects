from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from poetry50m.cli import main
from poetry50m.config import file_hash


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def test_general_binary_artifact_trains_and_writes_model_life(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    tokenizer = Tokenizer(
        WordLevel(
            {
                "<|pad|>": 0,
                "<|bos|>": 1,
                "<|eos|>": 2,
                "<|system|>": 3,
                "<|user|>": 4,
                "<|assistant|>": 5,
                "<|mask|>": 6,
                "word": 7,
            },
            unk_token="<|pad|>",
        )
    )
    tokenizer.save(str(prepared / "tokenizer.json"))
    rows = (np.arange(36, dtype="<u2") % 8).reshape(4, 9)
    for split in ("train", "validation", "test"):
        rows.tofile(prepared / f"{split}.tokens.bin")
    hashes = {
        "tokenizer.json": file_hash(prepared / "tokenizer.json"),
        **{
            f"{split}.tokens.bin": file_hash(prepared / f"{split}.tokens.bin")
            for split in ("train", "validation", "test")
        },
    }
    _write_json(
        prepared / "metadata.json",
        {
            "format_version": 2,
            "artifact_type": "general_pretraining_binary",
            "token_dtype": "uint16_le",
            "row_width": 9,
            "batch_size": 2,
            "tokenizer_hash": hashes["tokenizer.json"],
            "artifact_hashes": hashes,
            "config": {"batch_size": 2},
            "splits": {
                split: {"row_count": 4}
                for split in ("train", "validation", "test")
            },
        },
    )
    model_config = tmp_path / "model.json"
    _write_json(
        model_config,
        {
            "architecture": "gpt",
            "vocab_size": 8,
            "max_seq_len": 8,
            "d_model": 16,
            "n_layers": 1,
            "n_heads": 2,
            "ffn_dim": 32,
            "dropout": 0.0,
            "rope_base": 10000.0,
            "rope_fraction": 1.0,
            "norm_epsilon": 1e-6,
            "linear_bias": False,
            "tie_embeddings": True,
            "ignore_index": -100,
        },
    )
    train_config = tmp_path / "train.json"
    _write_json(
        train_config,
        {
            "max_steps": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "warmup_steps": 0,
            "device": "cpu",
            "precision": "none",
            "seed": 13,
            "deterministic": True,
            "log_every_steps": 1,
            "checkpoint_every_steps": 0,
            "trajectory_every_steps": 0,
            "trajectory_capture_steps": [1],
            "analysis_every_steps": 0,
        },
    )
    run = tmp_path / "run"

    assert (
        main(
            (
                "train",
                "--prepared",
                str(prepared),
                "--model-config",
                str(model_config),
                "--train-config",
                str(train_config),
                "--run-dir",
                str(run),
                "--batch-size",
                "2",
                "--data-seed",
                "17",
            )
        )
        == 0
    )

    assert (run / "trajectory" / "initial.pt").is_file()
    assert (run / "trajectory" / "step_00000001.pt").is_file()
    assert (run / "trajectory" / "final.pt").is_file()
    assert (run / "checkpoints" / "final.pt").is_file()
    assert (run / "run.manifest.json").is_file()
    assert (run / "train.receipt.json").is_file()
