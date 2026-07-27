"""Copy this file into the real Track 1 repository and replace the marked sections."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

from genome.adapters.base import Track1Adapter


class Poetry50MAdapter(Track1Adapter):
    adapter_id = "poetry_50m.track1.v1"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.project_config = config or {}

    def build_model(self) -> nn.Module:
        # Import and call the exact Track 1 constructor here. It must return W0 reproducibly.
        raise NotImplementedError("wire the exact Track 1 model constructor")

    def load_checkpoint(self, model: nn.Module, path: str | Path) -> None:
        # Use the exact Track 1 loader if its checkpoint schema is not a plain state_dict.
        super().load_checkpoint(model, path)

    def tokenizer_manifest(self) -> dict[str, Any]:
        # Return vocab size, special tokens, tokenizer hashes, normalization and source paths.
        raise NotImplementedError

    def corpus_manifest(self) -> dict[str, Any]:
        # Return pinned source manifests, prepared corpus IDs, dedup/split hashes and provenance.
        raise NotImplementedError

    def training_recipe(self) -> dict[str, Any]:
        # Return optimizer, schedule, precision, seeds, data order and software/environment IDs.
        raise NotImplementedError

    def split_manifest(self) -> dict[str, Any]:
        # Return source-document/poem-level IDs for fit/fingerprint/probe/development/hidden splits.
        raise NotImplementedError

    def evaluation_batches(self, split: str, max_batches: int | None = None) -> Iterable[Any]:
        # Yield the exact tokenized batches used by the declared evaluator split.
        raise NotImplementedError

    def batch_loss(self, model: nn.Module, batch: Any) -> tuple[torch.Tensor, int]:
        # The base implementation works for standard causal-LM logits and labels. Override only
        # when Track 1 uses a custom output/objective contract.
        return super().batch_loss(model, batch)


def create_adapter(config: dict[str, Any] | None = None) -> Poetry50MAdapter:
    return Poetry50MAdapter(config=config)
