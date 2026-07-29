#!/usr/bin/env python3
"""Validate and seal the review facts for the general 8M pretraining run."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from poetry50m.config import config_hash, file_hash, load_mapping
from poetry50m.data.binary_stream import (
    BinaryTokenBatchStream,
    load_binary_token_artifact,
)
from poetry50m.model import DecoderOnlyTransformer, ModelConfig
from poetry50m.training import TrainConfig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--data-seed", type=int, required=True)
    parser.add_argument("--expected-parameters", type=int, default=8_335_008)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.data_seed < 0:
        raise ValueError("data-seed must be non-negative")
    if args.output.exists():
        raise FileExistsError(f"pretrain validation output already exists: {args.output}")
    artifact = load_binary_token_artifact(args.prepared)
    model_config = ModelConfig.from_mapping(load_mapping(args.model_config))
    train_config = TrainConfig.from_mapping(load_mapping(args.train_config))
    model = DecoderOnlyTransformer(model_config)
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if parameter_count != args.expected_parameters:
        raise ValueError(
            f"model has {parameter_count:,} parameters, expected {args.expected_parameters:,}"
        )
    artifact_batch_size = artifact.metadata.get("batch_size")
    if artifact_batch_size != artifact.metadata["config"]["batch_size"]:
        raise ValueError("binary artifact batch-size metadata disagrees")
    batch_size = int(artifact_batch_size)
    train_rows = artifact.row_count("train")
    if train_rows % batch_size:
        raise ValueError("train rows do not form complete batches")
    one_epoch_steps = train_rows // batch_size
    if train_config.max_steps != one_epoch_steps:
        raise ValueError("training max_steps is not exactly one binary-artifact epoch")
    if train_config.seed == args.data_seed:
        raise ValueError("model and data seeds must be distinct for this run")
    stream = BinaryTokenBatchStream(
        artifact,
        batch_size=batch_size,
        seed=args.data_seed,
    )
    initial_stream_state = stream.state_dict()
    first_batch = next(stream)
    if first_batch["input_ids"].shape != (
        batch_size,
        model_config.max_seq_len,
    ):
        raise ValueError("first binary batch does not match the model context")
    receipt = {
        "format_version": 1,
        "mode": "general_8m_pretraining_review",
        "prepared": str(args.prepared),
        "prepared_metadata_sha256": file_hash(args.prepared / "metadata.json"),
        "tokenizer_sha256": artifact.metadata["tokenizer_hash"],
        "model_config": str(args.model_config),
        "model_config_sha256": file_hash(args.model_config),
        "model_config_identity": config_hash(model_config),
        "train_config": str(args.train_config),
        "train_config_sha256": file_hash(args.train_config),
        "train_config_identity": config_hash(train_config),
        "parameter_count": parameter_count,
        "model_seed": train_config.seed,
        "data_seed": args.data_seed,
        "batch_size": batch_size,
        "context_length": model_config.max_seq_len,
        "train_rows": train_rows,
        "one_epoch_steps": one_epoch_steps,
        "train_data_tokens": artifact.metadata["splits"]["train"]["data_token_count"],
        "source_rows": artifact.metadata["splits"]["train"]["source_rows"],
        "validation_data_tokens": artifact.metadata["splits"]["validation"][
            "data_token_count"
        ],
        "test_data_tokens": artifact.metadata["splits"]["test"]["data_token_count"],
        "initial_stream_state": initial_stream_state,
        "checkpoint_steps": list(train_config.checkpoint_steps),
        "trajectory_capture_steps": list(train_config.trajectory_capture_steps),
        "artifact_hashes": artifact.metadata["artifact_hashes"],
        "train_config_values": asdict(train_config),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
