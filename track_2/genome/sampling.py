from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .hashing import sha256_file
from .io import atomic_write_json


def prepare_dataset_sample(
    *,
    repository: str,
    revision: str,
    split: str,
    tokenizer_path: str | Path,
    output: str | Path,
    configuration: str | None = None,
    text_field: str = "text",
    examples: int = 4096,
    context_length: int = 2048,
    seed: int = 20260729,
    shuffle_buffer: int = 10_000,
) -> dict[str, Any]:
    """Stream a deterministic content sample and write raw/tokenized JSONL plus a receipt."""
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("datasets and transformers are required for sample preparation") from error
    dataset = load_dataset(
        repository,
        name=configuration,
        split=split,
        revision=revision,
        streaming=True,
    )
    dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    raw_path = root / "raw.jsonl"
    token_path = root / "tokens.jsonl"
    count = 0
    token_count = 0
    with raw_path.open("w", encoding="utf-8") as raw_handle, token_path.open(
        "w", encoding="utf-8"
    ) as token_handle:
        for row in dataset:
            text = row.get(text_field)
            if not isinstance(text, str) or not text.strip():
                continue
            ids = tokenizer(
                text,
                truncation=True,
                max_length=context_length,
                add_special_tokens=False,
            )["input_ids"]
            if len(ids) < 2:
                continue
            raw_handle.write(json.dumps({text_field: text}, ensure_ascii=False) + "\n")
            token_handle.write(json.dumps({"input_ids": ids}, separators=(",", ":")) + "\n")
            count += 1
            token_count += len(ids)
            if count >= examples:
                break
    if count < examples:
        raise ValueError(f"dataset stream yielded only {count} usable examples; requested {examples}")
    receipt = {
        "format": "GENOME_DATASET_SAMPLE",
        "version": "1.0.0",
        "repository": repository,
        "revision": revision,
        "configuration": configuration,
        "split": split,
        "text_field": text_field,
        "examples": count,
        "tokens": token_count,
        "context_length": context_length,
        "seed": seed,
        "shuffle_buffer": shuffle_buffer,
        "raw": {"path": str(raw_path), "bytes": raw_path.stat().st_size, "sha256": sha256_file(raw_path)},
        "tokens_file": {
            "path": str(token_path),
            "bytes": token_path.stat().st_size,
            "sha256": sha256_file(token_path),
        },
    }
    atomic_write_json(root / "receipt.json", receipt)
    return receipt
