#!/usr/bin/env python3
"""Generate a poem from the final Track 1 checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

from poetry50m.inference import GenerationConfig, generate
from poetry50m.model import DecoderOnlyTransformer, ModelConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "runs/full-pretrain-v4-b32/checkpoints/final.pt"
DEFAULT_TOKENIZER = PROJECT_ROOT / "artifacts/prepared_v3/tokenizer.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a poem from the final 8M Track 1 model.",
    )
    parser.add_argument("prompt", help='Prompt such as "Write a poem about moonlit water."')
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> int:
    args = parse_args()
    device = select_device(args.device)
    checkpoint = torch.load(
        args.checkpoint.expanduser(),
        map_location="cpu",
        weights_only=False,
    )
    model = DecoderOnlyTransformer(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)

    tokenizer = Tokenizer.from_file(str(args.tokenizer.expanduser()))
    result = generate(
        model,
        tokenizer,
        args.prompt,
        GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
        ),
    )
    print(result.generated_text)
    print(
        f"[{result.stop_reason}; {result.generated_token_count} tokens; "
        f"{result.wall_seconds:.2f}s on {device}]",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
