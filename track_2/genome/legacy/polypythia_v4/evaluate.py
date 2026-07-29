"""Legacy adapters for PolyPythia V4 shared-decoder evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ...neural.multilife_decoder import load_shared_decoder
from ...polypythia.evaluate import (
    evaluate_shared_decoder_corpus as _evaluate_shared_decoder_corpus,
)
from ...polypythia.evaluate import execute_hidden_prediction as _execute_hidden_prediction
from ...polypythia.lives import CanonicalModelLife
from ...types import TensorSpec


def execute_hidden_prediction(
    hidden_life: CanonicalModelLife,
    *,
    tensor_specs: Sequence[TensorSpec],
    tied_groups: Sequence[Sequence[str]],
    shared_decoder_path: str | Path,
    prediction_path: str | Path,
    config_path: str | Path,
    output_path: str | Path,
    device: str = "cuda",
) -> dict[str, object]:
    return _execute_hidden_prediction(
        hidden_life,
        tensor_specs=tensor_specs,
        tied_groups=tied_groups,
        shared_decoder_path=shared_decoder_path,
        prediction_path=prediction_path,
        config_path=config_path,
        output_path=output_path,
        device=device,
        shared_decoder_loader=load_shared_decoder,
    )


def evaluate_shared_decoder_corpus(
    *,
    training_lives: Sequence[CanonicalModelLife],
    development_life: CanonicalModelLife,
    tensor_specs: Sequence[TensorSpec],
    tied_groups: Sequence[Sequence[str]],
    shared_decoder_path: str | Path,
    development_code_path: str | Path,
    config_path: str | Path,
    tokenizer_path: str | Path,
    evaluation_texts_path: str | Path,
    output_path: str | Path,
    device: str = "cuda",
    sequence_length: int = 512,
    batch_size: int = 4,
    max_batches: int | None = None,
    anchors_per_batch: int = 8,
) -> dict[str, object]:
    return _evaluate_shared_decoder_corpus(
        training_lives=training_lives,
        development_life=development_life,
        tensor_specs=tensor_specs,
        tied_groups=tied_groups,
        shared_decoder_path=shared_decoder_path,
        development_code_path=development_code_path,
        config_path=config_path,
        tokenizer_path=tokenizer_path,
        evaluation_texts_path=evaluation_texts_path,
        output_path=output_path,
        device=device,
        sequence_length=sequence_length,
        batch_size=batch_size,
        max_batches=max_batches,
        anchors_per_batch=anchors_per_batch,
        shared_decoder_loader=load_shared_decoder,
    )
