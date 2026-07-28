"""Validated training access to a finalized SFT mixture."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from tokenizers import Tokenizer

from poetry50m.config import file_hash

from .artifacts import read_packed_sequences
from .batch_stream import PreparedBatchStream
from .packing import PackedSequence
from .schema import ObjectiveMix


@dataclass(frozen=True, slots=True)
class SftTrainingArtifact:
    root: Path
    receipt: dict[str, Any]
    packs: tuple[PackedSequence, ...]

    @property
    def packs_path(self) -> Path:
        return self.root / cast(str, self.receipt["packs_filename"])


def _read_receipt(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError("SFT mixture receipt must be a JSON object")
    return cast(dict[str, Any], value)


def _required_integer(receipt: dict[str, Any], name: str) -> int:
    value = receipt.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"SFT mixture receipt has invalid {name}")
    return value


def _artifact_path(root: Path, receipt: dict[str, Any], name: str) -> Path:
    filename = receipt.get(name)
    if (
        not isinstance(filename, str)
        or not filename
        or Path(filename).name != filename
    ):
        raise ValueError(f"SFT mixture receipt has invalid {name}")
    return root / filename


def load_sft_training_artifact(
    root: Path,
    *,
    tokenizer_path: Path,
) -> SftTrainingArtifact:
    """Validate the sealed mixture and return its already-packed training rows."""
    receipt = _read_receipt(root / "receipt.json")
    if receipt.get("format_version") != 1:
        raise ValueError("unsupported SFT mixture receipt")
    dataset_path = _artifact_path(root, receipt, "dataset_filename")
    packs_path = _artifact_path(root, receipt, "packs_filename")
    if file_hash(dataset_path) != receipt.get("dataset_sha256"):
        raise ValueError("SFT dataset hash does not match its receipt")
    if file_hash(packs_path) != receipt.get("packs_sha256"):
        raise ValueError("SFT packs hash does not match their receipt")
    if file_hash(tokenizer_path) != receipt.get("tokenizer_sha256"):
        raise ValueError("SFT tokenizer hash does not match the mixture receipt")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    sequence_length = _required_integer(receipt, "sequence_length")
    packs = read_packed_sequences(packs_path)
    if len(packs) != _required_integer(receipt, "pack_count"):
        raise ValueError("SFT pack count does not match its receipt")
    if {pack.pack_id for pack in packs} != set(range(len(packs))):
        raise ValueError("SFT pack IDs must be unique and contiguous")
    if any(
        pack.boundary_key != "sft"
        or pack.objective != "conditional_poetry"
        or len(pack.input_ids) > sequence_length
        or max(pack.input_ids) >= tokenizer.get_vocab_size()
        for pack in packs
    ):
        raise ValueError("SFT packs violate the training contract")
    formatted_tokens = sum(len(pack.input_ids) for pack in packs)
    supervised_tokens = sum(sum(pack.loss_mask) for pack in packs)
    if formatted_tokens != _required_integer(receipt, "actual_formatted_tokens"):
        raise ValueError("SFT formatted-token total does not match its receipt")
    if supervised_tokens != _required_integer(receipt, "supervised_tokens"):
        raise ValueError("SFT supervised-token total does not match its receipt")
    return SftTrainingArtifact(root=root, receipt=receipt, packs=packs)


def sft_batch_stream(
    artifact: SftTrainingArtifact,
    *,
    batch_size: int,
    pad_token_id: int,
    seed: int,
) -> PreparedBatchStream:
    return PreparedBatchStream.from_artifact(
        str(artifact.packs_path),
        batch_size=batch_size,
        pad_token_id=pad_token_id,
        objective_mix=ObjectiveMix(
            conditional_poetry=1.0,
            auxiliary_prose_ntp=0.0,
            poetry_ntp=0.0,
        ),
        seed=seed,
    )
