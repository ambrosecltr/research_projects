from __future__ import annotations

import json
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from poetry50m.cli import main
from poetry50m.config import file_hash
from poetry50m.data.artifacts import write_packed_sequences
from poetry50m.data.packing import PackedSequence
from poetry50m.model import DecoderOnlyTransformer, ModelConfig
from poetry50m.training import CyclingBatchStream, TrainConfig, Trainer
from poetry50m.trajectory.manifest import RunManifest


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def test_sft_cli_starts_from_pretrained_weights_and_resumes(tmp_path: Path) -> None:
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
                "light": 7,
            },
            unk_token="<|pad|>",
        )
    ).save(str(tokenizer_path))
    model_mapping = {
        "architecture": "gpt",
        "vocab_size": 8,
        "max_seq_len": 8,
        "d_model": 8,
        "n_layers": 1,
        "n_heads": 2,
        "ffn_dim": 16,
        "dropout": 0.0,
        "rope_base": 10000.0,
        "rope_fraction": 1.0,
        "norm_epsilon": 1e-6,
        "linear_bias": False,
        "tie_embeddings": True,
        "ignore_index": -100,
    }
    _json(tmp_path / "model.json", model_mapping)
    train_mapping = {
        "max_steps": 2,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "warmup_steps": 0,
        "device": "cpu",
        "precision": "none",
        "seed": 17,
        "deterministic": True,
        "log_every_steps": 1,
        "checkpoint_every_steps": 0,
        "trajectory_every_steps": 0,
        "analysis_every_steps": 0,
    }
    _json(tmp_path / "sft-train.json", train_mapping)

    base_directory = tmp_path / "base"
    base_model = DecoderOnlyTransformer(ModelConfig.from_mapping(model_mapping))
    base_trainer = Trainer(
        base_model,
        TrainConfig(
            max_steps=1,
            learning_rate=0.001,
            weight_decay=0.0,
            device="cpu",
            precision="none",
        ),
        base_directory,
        run_metadata={"run_id": "base-run"},
    )
    base_stream = CyclingBatchStream(
        [
            {
                "input_ids": torch.tensor([[1, 3, 5, 4, 6]]),
                "targets": torch.tensor([[3, 5, 4, 6, 2]]),
                "loss_mask": torch.tensor([[False, False, False, True, True]]),
                "data_token_count": 5,
            }
        ]
    )
    base_trainer.fit(base_stream)
    base_checkpoint = base_trainer.save_checkpoint(base_directory / "final.pt")
    base_manifest = RunManifest(
        run_id="base-run",
        initialization_id="base-initialization",
        data_order_id="base-order",
        architecture_signature="gpt",
        corpus_signature="base-corpus",
        tokenizer_hash=file_hash(tokenizer_path),
        code_signature="base-code",
        model_config_hash=base_trainer.model_config_hash,
        training_config_hash=base_trainer.train_config_hash,
        endpoint_sealed=False,
    )
    base_manifest.save(base_directory / "run.manifest.json")
    _json(
        base_directory / "train.receipt.json",
        {
            "checkpoint_sha256": file_hash(base_checkpoint),
            "run_id": base_manifest.run_id,
        },
    )

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
                input_ids=(1, 3, 5, 4, 6, 7, 2),
                loss_mask=(False, False, False, False, True, True, True),
            ),
        ),
    )
    _json(
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
            "actual_formatted_tokens": 7,
            "supervised_tokens": 3,
        },
    )
    run_directory = tmp_path / "sft-run"
    common = (
        "--mixture",
        str(mixture),
        "--tokenizer",
        str(tokenizer_path),
        "--base-checkpoint",
        str(base_checkpoint),
        "--base-manifest",
        str(base_directory / "run.manifest.json"),
        "--base-receipt",
        str(base_directory / "train.receipt.json"),
        "--model-config",
        str(tmp_path / "model.json"),
        "--train-config",
        str(tmp_path / "sft-train.json"),
        "--batch-size",
        "1",
    )
    plan_path = tmp_path / "sft-plan.json"
    assert main(("sft-validate", *common, "--output", str(plan_path))) == 0
    assert json.loads(plan_path.read_text(encoding="utf-8"))["planned_epochs"] == 2

    train_common = (*common, "--run-dir", str(run_directory))
    assert main(("sft-train", *train_common, "--until-step", "1")) == 0
    checkpoint = run_directory / "checkpoints" / "final.pt"
    first = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert first["training_state"]["global_step"] == 1
    assert first["training_state"]["supervised_tokens_seen"] == 3

    assert main(("sft-train", *train_common, "--resume", str(checkpoint))) == 0
    resumed = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert resumed["training_state"]["global_step"] == 2
    assert resumed["training_state"]["supervised_tokens_seen"] == 6
