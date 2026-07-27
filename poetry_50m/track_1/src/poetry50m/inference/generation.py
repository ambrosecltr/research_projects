"""Strict, deterministic conditional generation for Track 1 language models."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
from tokenizers import Tokenizer
from torch import Tensor

from poetry50m.data import reserved_token_ids
from poetry50m.data.tokenizer import SPECIAL_TOKENS
from poetry50m.model import DecoderOnlyTransformer
from poetry50m.training import mapping_hash
from poetry50m.trajectory.snapshots import assert_identical_coordinates, load_weight_snapshot
from poetry50m.trajectory.types import SnapshotMetadata, WeightSnapshot

StopReason = Literal["eos", "max_new_tokens"]


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Validated sampling settings; a seed is required for reproducible samples."""

    max_new_tokens: int
    temperature: float
    top_p: float
    seed: int

    def __post_init__(self) -> None:
        if isinstance(self.max_new_tokens, bool) or not isinstance(self.max_new_tokens, int):
            raise TypeError("max_new_tokens must be an integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        for name, value in (("temperature", self.temperature), ("top_p", self.top_p)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a finite number")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must lie in (0, 1]")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Generated poem continuation excluding the EOS terminator, when present."""

    conditioning_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    generated_text: str
    stop_reason: StopReason
    wall_seconds: float

    def __post_init__(self) -> None:
        for name, token_ids in (
            ("conditioning_token_ids", self.conditioning_token_ids),
            ("generated_token_ids", self.generated_token_ids),
        ):
            if not isinstance(token_ids, tuple):
                raise TypeError(f"{name} must be a tuple")
            if any(
                isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
                for token_id in token_ids
            ):
                raise ValueError(f"{name} must contain non-negative integer token IDs")
        if not self.conditioning_token_ids:
            raise ValueError("conditioning_token_ids must not be empty")
        if self.stop_reason not in {"eos", "max_new_tokens"}:
            raise ValueError("invalid generation stop reason")
        if isinstance(self.wall_seconds, bool) or not isinstance(self.wall_seconds, (int, float)):
            raise TypeError("wall_seconds must be a finite number")
        if not math.isfinite(self.wall_seconds):
            raise ValueError("wall_seconds must be finite")
        if self.wall_seconds < 0.0:
            raise ValueError("wall_seconds cannot be negative")

    @property
    def generated_token_count(self) -> int:
        return len(self.generated_token_ids)


def _special_token_ids(tokenizer: Tokenizer) -> dict[str, int]:
    ids = {token: tokenizer.token_to_id(token) for token in SPECIAL_TOKENS}
    missing = [token for token, token_id in ids.items() if token_id is None]
    if missing:
        raise ValueError(f"tokenizer lacks required special tokens: {missing}")
    resolved: dict[str, int] = {}
    for token, token_id in ids.items():
        if token_id is None:
            raise AssertionError("missing special token was not reported")
        resolved[token] = token_id
    return resolved


def build_conditioning_tokens(
    tokenizer: Tokenizer,
    prompt: str,
    thought: str | None = None,
) -> tuple[int, ...]:
    """Encode the exact training prefix: BOS, prompt, optional thought, poem."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if thought is not None and (not isinstance(thought, str) or not thought.strip()):
        raise ValueError("thought must be a non-empty string when supplied")
    ids = _special_token_ids(tokenizer)
    prefix = [ids["<|bos|>"], ids["<|prompt|>"]]
    prefix.extend(tokenizer.encode(prompt, add_special_tokens=False).ids)
    if thought is not None:
        prefix.append(ids["<|thought|>"])
        prefix.extend(tokenizer.encode(thought, add_special_tokens=False).ids)
    prefix.append(ids["<|poem|>"])
    return tuple(prefix)


def _sample_top_p(logits: Tensor, config: GenerationConfig, generator: torch.Generator) -> int:
    probabilities = torch.softmax((logits.float() / config.temperature).cpu(), dim=-1)
    sorted_probabilities, sorted_indices = torch.sort(probabilities, descending=True)
    cumulative = torch.cumsum(sorted_probabilities, dim=-1)
    keep = cumulative - sorted_probabilities < config.top_p
    keep[0] = True
    filtered = sorted_probabilities * keep
    filtered = filtered / filtered.sum()
    selected = torch.multinomial(filtered, 1, generator=generator)
    return int(sorted_indices[selected].item())


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def generate(
    model: DecoderOnlyTransformer,
    tokenizer: Tokenizer,
    prompt: str,
    config: GenerationConfig,
    *,
    thought: str | None = None,
) -> GenerationResult:
    """Generate one poem continuation without allowing structural control tokens."""
    tokenizer_vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    if tokenizer_vocab_size != model.config.vocab_size:
        raise ValueError(
            "tokenizer vocabulary size must exactly match model.config.vocab_size "
            f"({tokenizer_vocab_size} != {model.config.vocab_size})"
        )
    device = next(model.parameters()).device
    _synchronize_device(device)
    started_at = time.perf_counter()
    was_training = model.training
    model.eval()
    try:
        conditioning = build_conditioning_tokens(tokenizer, prompt, thought)
        if len(conditioning) + config.max_new_tokens > model.config.max_seq_len:
            raise ValueError(
                "conditioning length plus max_new_tokens exceeds the model context; "
                "reduce the request"
            )
        special_ids = _special_token_ids(tokenizer)
        eos_id = special_ids["<|eos|>"]
        reserved_ids = reserved_token_ids(tokenizer)
        if any(
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or not 0 <= token_id < model.config.vocab_size
            for token_id in reserved_ids
        ):
            raise ValueError("reserved tokenizer IDs must lie within the model vocabulary")
        suppressed_ids = sorted(
            {token_id for token, token_id in special_ids.items() if token != "<|eos|>"}.union(
                reserved_ids
            )
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.seed)
        generated: list[int] = []
        stop_reason: StopReason = "max_new_tokens"
        with torch.inference_mode():
            inputs = torch.tensor([conditioning], dtype=torch.long, device=device)
            cached = model.forward_cached(inputs)
            for token_index in range(config.max_new_tokens):
                next_logits = cached.logits[0, -1].clone()
                next_logits[suppressed_ids] = -torch.inf
                token_id = _sample_top_p(next_logits, config, generator)
                if token_id == eos_id:
                    stop_reason = "eos"
                    break
                generated.append(token_id)
                if token_index + 1 < config.max_new_tokens:
                    cached = model.forward_cached(
                        torch.tensor([[token_id]], dtype=torch.long, device=device),
                        cached.cache,
                    )
    finally:
        try:
            _synchronize_device(device)
        finally:
            model.train(was_training)
    wall_seconds = time.perf_counter() - started_at
    return GenerationResult(
        conditioning_token_ids=conditioning,
        generated_token_ids=tuple(generated),
        generated_text=tokenizer.decode(generated, skip_special_tokens=False),
        stop_reason=stop_reason,
        wall_seconds=wall_seconds,
    )


def load_snapshot_into_model(
    model: DecoderOnlyTransformer,
    snapshot_path: Path,
    *,
    expected_metadata: SnapshotMetadata | None = None,
) -> SnapshotMetadata:
    """Apply a local restricted weight snapshot only after strict coordinate checks."""
    snapshot = load_weight_snapshot(snapshot_path)
    if expected_metadata is not None and snapshot.metadata != expected_metadata:
        raise ValueError("snapshot metadata does not match the expected sealed checkpoint")
    if snapshot.metadata.model_config_hash != mapping_hash(asdict(model.config)):
        raise ValueError("snapshot model configuration does not match the live model")
    expected = WeightSnapshot(metadata=snapshot.metadata, state_dict=model.state_dict())
    assert_identical_coordinates(expected, snapshot)
    model.load_state_dict(snapshot.state_dict, strict=True)
    return snapshot.metadata
