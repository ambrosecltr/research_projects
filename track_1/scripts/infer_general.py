#!/usr/bin/env python3
"""Generate one response from a general Track 1 checkpoint."""

from __future__ import annotations

import argparse
import pickle
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import torch
from tokenizers import Tokenizer
from torch import Tensor

from poetry50m.config import load_mapping
from poetry50m.data import reserved_token_ids
from poetry50m.model import DecoderOnlyTransformer, ModelConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "runs/general_8m/sft/full-v2/checkpoints/final.pt"
)
DEFAULT_TOKENIZER = PROJECT_ROOT / "artifacts/general_8m/pretrain_v2/tokenizer.json"
SPECIAL_TOKENS = (
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|mask|>",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument(
        "--model-config",
        type=Path,
        default=PROJECT_ROOT / "configs/model/general_8m.yaml",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return device


def checkpoint_weights(path: Path) -> Mapping[str, Any]:
    try:
        payload: object = torch.load(path, map_location="cpu", weights_only=True)
    except (EOFError, OSError, pickle.UnpicklingError, RuntimeError, ValueError) as error:
        raise ValueError("checkpoint must be a restricted Track 1 checkpoint") from error
    if not isinstance(payload, Mapping) or payload.get("format_version") != 2:
        raise ValueError("unsupported Track 1 checkpoint")
    model = payload.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("checkpoint lacks model weights")
    return cast(Mapping[str, Any], model)


def sample_top_p(
    logits: Tensor,
    *,
    temperature: float,
    top_p: float,
    generator: torch.Generator,
) -> int:
    probabilities = torch.softmax((logits.float() / temperature).cpu(), dim=-1)
    sorted_probabilities, sorted_indices = torch.sort(probabilities, descending=True)
    cumulative = torch.cumsum(sorted_probabilities, dim=-1)
    keep = cumulative - sorted_probabilities < top_p
    keep[0] = True
    filtered = sorted_probabilities * keep
    selected = torch.multinomial(filtered / filtered.sum(), 1, generator=generator)
    return int(sorted_indices[selected].item())


def main() -> int:
    args = parse_args()
    if args.seed < 0 or args.max_new_tokens < 1:
        raise ValueError("seed and max-new-tokens are invalid")
    if args.temperature <= 0.0 or not 0.0 < args.top_p <= 1.0:
        raise ValueError("temperature or top-p is invalid")

    tokenizer = Tokenizer.from_file(str(args.tokenizer.expanduser()))
    token_ids = {token: tokenizer.token_to_id(token) for token in SPECIAL_TOKENS}
    if any(token_id is None for token_id in token_ids.values()):
        raise ValueError("tokenizer lacks a required general-model special token")
    ids = cast(dict[str, int], token_ids)

    config = ModelConfig.from_mapping(load_mapping(args.model_config.expanduser()))
    if tokenizer.get_vocab_size(with_added_tokens=True) != config.vocab_size:
        raise ValueError("tokenizer vocabulary does not match the model configuration")
    model = DecoderOnlyTransformer(config)
    model.load_state_dict(dict(checkpoint_weights(args.checkpoint.expanduser())), strict=True)
    device = select_device(args.device)
    model.to(device).eval()

    conditioning = [
        ids["<|bos|>"],
        ids["<|user|>"],
        *tokenizer.encode(args.prompt, add_special_tokens=False).ids,
        ids["<|assistant|>"],
    ]
    if len(conditioning) + args.max_new_tokens > config.max_seq_len:
        raise ValueError("prompt and requested response exceed the model context")
    suppressed = sorted(
        {
            token_id
            for token, token_id in ids.items()
            if token != "<|eos|>"
        }.union(reserved_token_ids(tokenizer))
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    generated: list[int] = []
    stop_reason = "max_new_tokens"
    with torch.inference_mode():
        cached = model.forward_cached(
            torch.tensor([conditioning], dtype=torch.long, device=device)
        )
        for index in range(args.max_new_tokens):
            logits = cached.logits[0, -1].clone()
            logits[suppressed] = -torch.inf
            token_id = sample_top_p(
                logits,
                temperature=args.temperature,
                top_p=args.top_p,
                generator=generator,
            )
            if token_id == ids["<|eos|>"]:
                stop_reason = "eos"
                break
            generated.append(token_id)
            if index + 1 < args.max_new_tokens:
                cached = model.forward_cached(
                    torch.tensor([[token_id]], dtype=torch.long, device=device),
                    cached.cache,
                )

    print(tokenizer.decode(generated, skip_special_tokens=False))
    print(f"[{stop_reason}; {len(generated)} tokens; {device}]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
