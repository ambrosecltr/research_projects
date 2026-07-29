from __future__ import annotations

import json
import random
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from urllib.request import Request, urlopen

import numpy as np

from .hashing import sha256_file
from .io import atomic_write_json


def _read_byte_range(url: str, *, start: int, end: int, total: int) -> bytes:
    request = Request(
        url,
        headers={
            "Range": f"bytes={start}-{end}",
            "User-Agent": "genome-track2/1.0",
        },
    )
    with urlopen(request, timeout=120) as response:
        payload = response.read()
        status = response.status
        content_range = response.headers.get("Content-Range")
    expected_bytes = end - start + 1
    if status != 206:
        raise ValueError(f"dataset server ignored byte range with HTTP {status}")
    if content_range != f"bytes {start}-{end}/{total}":
        raise ValueError(f"unexpected dataset Content-Range: {content_range!r}")
    if len(payload) != expected_bytes:
        raise ValueError(
            f"dataset byte range returned {len(payload)} bytes; expected {expected_bytes}"
        )
    return payload


def prepare_dataset_sample(
    *,
    repository: str,
    revision: str,
    tokenizer_path: str | Path,
    output: str | Path,
    filename: str = "document-00000-of-00020.bin",
    examples: int = 4096,
    context_length: int = 2048,
    seed: int = 20260729,
) -> dict[str, Any]:
    """Read an aligned exact-token sample from the official Pythia Pile binary."""
    try:
        from huggingface_hub import HfApi
        from transformers import AutoTokenizer
    except ImportError as error:  # pragma: no cover
        raise RuntimeError(
            "huggingface-hub and transformers are required for sample preparation"
        ) from error
    if examples <= 0:
        raise ValueError("examples must be positive")
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    info = HfApi().dataset_info(repository, revision=revision, files_metadata=True)
    if not info.sha:
        raise ValueError("Hugging Face did not return an immutable dataset commit")
    sibling = next((item for item in info.siblings if item.rfilename == filename), None)
    if sibling is None or sibling.size is None:
        raise ValueError(f"dataset file metadata is unavailable for {filename}")
    source_sequence_length = context_length + 1
    bytes_per_sequence = source_sequence_length * np.dtype("<u2").itemsize
    available_sequences = sibling.size // bytes_per_sequence
    if examples > available_sequences:
        raise ValueError(
            f"requested {examples} examples from a shard with {available_sequences} aligned sequences"
        )
    start_sequence = random.Random(seed).randrange(available_sequences - examples + 1)
    byte_start = start_sequence * bytes_per_sequence
    byte_end = byte_start + examples * bytes_per_sequence - 1
    url = (
        f"https://huggingface.co/datasets/{repository}/resolve/"
        f"{info.sha}/{filename}"
    )
    payload = _read_byte_range(
        url,
        start=byte_start,
        end=byte_end,
        total=sibling.size,
    )
    sequences = np.frombuffer(payload, dtype="<u2").reshape(examples, source_sequence_length)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    root = Path(output)
    if root.exists():
        raise FileExistsError(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{root.name}-", dir=root.parent) as staging_value:
        staging = Path(staging_value)
        raw_path = staging / "raw.jsonl"
        token_path = staging / "tokens.jsonl"
        range_path = staging / "source-range.bin"
        range_path.write_bytes(payload)
        maximum_token_id = 0
        with raw_path.open("w", encoding="utf-8") as raw_handle, token_path.open(
            "w", encoding="utf-8"
        ) as token_handle:
            for sequence in sequences:
                ids = sequence[:context_length].astype(np.int64).tolist()
                maximum_token_id = max(maximum_token_id, max(ids))
                text = tokenizer.decode(
                    ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                raw_handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                token_handle.write(
                    json.dumps({"input_ids": ids}, separators=(",", ":")) + "\n"
                )
        if maximum_token_id >= len(tokenizer):
            raise ValueError(
                f"sample token ID {maximum_token_id} exceeds tokenizer size {len(tokenizer)}"
            )
        receipt = {
            "format": "GENOME_DATASET_SAMPLE",
            "version": "1.0.0",
            "repository": repository,
            "requested_revision": revision,
            "resolved_commit": info.sha,
            "filename": filename,
            "source_file_bytes": sibling.size,
            "source_dtype": "uint16-little-endian",
            "source_sequence_length": source_sequence_length,
            "start_sequence": start_sequence,
            "byte_range": {"start": byte_start, "end": byte_end},
            "examples": examples,
            "tokens": examples * context_length,
            "context_length": context_length,
            "seed": seed,
            "maximum_token_id": maximum_token_id,
            "raw_source": "tokenizer_decode_of_exact_training_tokens",
            "source_range": {
                "path": str(root / range_path.name),
                "bytes": range_path.stat().st_size,
                "sha256": sha256_file(range_path),
            },
            "raw": {
                "path": str(root / raw_path.name),
                "bytes": raw_path.stat().st_size,
                "sha256": sha256_file(raw_path),
            },
            "tokens_file": {
                "path": str(root / token_path.name),
                "bytes": token_path.stat().st_size,
                "sha256": sha256_file(token_path),
            },
        }
        atomic_write_json(staging / "receipt.json", receipt)
        staging.rename(root)
    return receipt
