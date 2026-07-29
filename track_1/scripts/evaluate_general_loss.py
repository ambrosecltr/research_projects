#!/usr/bin/env python3
"""Measure fixed held-out loss for a general 8M pretraining checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import torch

from poetry50m.config import file_hash, load_mapping
from poetry50m.data.binary_stream import (
    BinaryTokenBatchStream,
    load_binary_token_artifact,
)
from poetry50m.model import DecoderOnlyTransformer, ModelConfig


def _checkpoint_model(path: Path) -> Mapping[str, Any]:
    try:
        value: object = torch.load(path, map_location="cpu", weights_only=True)
    except (EOFError, OSError, pickle.UnpicklingError, RuntimeError, ValueError) as error:
        raise ValueError("checkpoint must be a restricted Track 1 checkpoint") from error
    if not isinstance(value, Mapping) or value.get("format_version") != 2:
        raise ValueError("unsupported Track 1 checkpoint")
    model = value.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("checkpoint lacks model weights")
    return cast(Mapping[str, Any], model)


def _device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--batches", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1206301356)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args()
    if args.batch_size < 1 or args.batches < 1 or args.seed < 0:
        raise ValueError("batch-size, batches, and seed are invalid")
    artifact = load_binary_token_artifact(args.prepared)
    available_batches = artifact.row_count(args.split) // args.batch_size
    if args.batches > available_batches:
        raise ValueError(
            f"requested {args.batches} batches but {args.split} has {available_batches}"
        )
    stream = BinaryTokenBatchStream(
        artifact,
        batch_size=args.batch_size,
        seed=args.seed,
        split=args.split,
    )
    config = ModelConfig.from_mapping(load_mapping(args.model_config))
    model = DecoderOnlyTransformer(config)
    model.load_state_dict(dict(_checkpoint_model(args.checkpoint)), strict=True)
    device = _device(args.device)
    model.to(device)
    model.eval()
    weighted_loss = 0.0
    token_count = 0
    with torch.inference_mode():
        for _ in range(args.batches):
            batch = next(stream)
            inputs = batch["input_ids"].to(device)
            targets = batch["targets"].to(device)
            loss_mask = batch["loss_mask"].to(device)
            output = model(inputs, targets, loss_mask)
            if output.loss is None:
                raise RuntimeError("model did not return held-out loss")
            weight = int(loss_mask.sum().item())
            weighted_loss += float(output.loss.item()) * weight
            token_count += weight
    mean_loss = weighted_loss / token_count
    receipt = {
        "format_version": 1,
        "mode": "general_pretraining_heldout_loss",
        "split": args.split,
        "batch_size": args.batch_size,
        "batches": args.batches,
        "seed": args.seed,
        "supervised_tokens": token_count,
        "mean_loss": mean_loss,
        "perplexity": math.exp(mean_loss),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "prepared_metadata_sha256": file_hash(args.prepared / "metadata.json"),
        "model_config_sha256": file_hash(args.model_config),
        "device": device.type,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"evaluation output already exists: {args.output}")
    args.output.write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
