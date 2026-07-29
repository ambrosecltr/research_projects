from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, Mapping

import torch


def token_sequences_from_jsonl(path: str | Path) -> Iterator[list[int]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            raw = value.get("input_ids")
            if not isinstance(raw, list) or any(not isinstance(item, int) for item in raw):
                raise ValueError(f"input_ids must be an integer array at line {line_number}")
            yield raw


def raw_texts_from_jsonl(path: str | Path, *, field: str = "text") -> Iterator[str]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            text = value.get(field)
            if not isinstance(text, str):
                raise ValueError(f"{field!r} must be a string at line {line_number}")
            yield text


def causal_batches_from_jsonl(path: str | Path) -> Iterator[dict[str, torch.Tensor]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            raw = value.get("input_ids")
            if not isinstance(raw, list) or len(raw) < 2 or any(not isinstance(item, int) for item in raw):
                raise ValueError(f"input_ids must contain at least two integers at line {line_number}")
            input_ids = torch.tensor([raw], dtype=torch.long)
            labels = input_ids.clone()
            yield {"input_ids": input_ids, "labels": labels}


def write_tokenized_jsonl(
    texts: Iterable[str],
    tokenizer,
    *,
    path: str | Path,
    context_length: int,
    limit: int | None = None,
) -> int:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("w", encoding="utf-8") as handle:
        for text in texts:
            if limit is not None and count >= limit:
                break
            encoded = tokenizer(text, truncation=True, max_length=context_length, add_special_tokens=False)
            ids = encoded["input_ids"]
            if len(ids) < 2:
                continue
            handle.write(json.dumps({"input_ids": ids}, separators=(",", ":")) + "\n")
            count += 1
    return count
