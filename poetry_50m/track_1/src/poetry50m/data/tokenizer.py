"""Deterministic byte-fallback BPE and conditional sequence encoding."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

from .schema import ConditionalExample, ProseNTPExample, TokenSequence

SPECIAL_TOKENS = (
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|prompt|>",
    "<|thought|>",
    "<|poem|>",
    "<|mask|>",
)
RESERVED_TOKEN_PREFIX = "<|reserved_"


@dataclass(frozen=True, slots=True)
class TokenizerSpec:
    vocab_size: int = 8_192
    min_frequency: int = 2
    special_tokens: tuple[str, ...] = SPECIAL_TOKENS

    def __post_init__(self) -> None:
        minimum_vocab = len(set(self.special_tokens)) + len(pre_tokenizers.ByteLevel.alphabet())
        if (
            isinstance(self.vocab_size, bool)
            or not isinstance(self.vocab_size, int)
            or self.vocab_size < minimum_vocab
        ):
            raise ValueError(
                f"vocab_size must accommodate byte alphabet and specials ({minimum_vocab})"
            )
        if (
            isinstance(self.min_frequency, bool)
            or not isinstance(self.min_frequency, int)
            or self.min_frequency < 1
        ):
            raise ValueError("min_frequency must be positive")
        if len(self.special_tokens) != len(set(self.special_tokens)):
            raise ValueError("special tokens must be unique")
        if self.special_tokens != SPECIAL_TOKENS:
            raise ValueError("special_tokens must exactly match the required ordered contract")


def train_tokenizer(texts: Iterable[str], spec: TokenizerSpec | None = None) -> Tokenizer:
    """Train reproducibly in supplied order with whitespace/newline-safe byte encoding."""
    spec = spec or TokenizerSpec()
    corpus = tuple(text.replace("\r\n", "\n").replace("\r", "\n") for text in texts)
    if not corpus or not any(text for text in corpus):
        raise ValueError("tokenizer training requires non-empty text")
    tokenizer = Tokenizer(models.BPE(unk_token=None, byte_fallback=True))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    trainer_factory = cast(Callable[..., trainers.BpeTrainer], trainers.BpeTrainer)
    trainer = trainer_factory(
        vocab_size=spec.vocab_size,
        min_frequency=spec.min_frequency,
        special_tokens=list(spec.special_tokens),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(corpus, trainer=trainer, length=len(corpus))
    for token in spec.special_tokens:
        if tokenizer.token_to_id(token) is None:
            raise RuntimeError(f"tokenizer failed to retain special token {token}")
    actual_vocab = tokenizer.get_vocab_size(with_added_tokens=True)
    if actual_vocab < spec.vocab_size:
        tokenizer.add_tokens(
            [
                f"{RESERVED_TOKEN_PREFIX}{index:05d}|>"
                for index in range(spec.vocab_size - actual_vocab)
            ]
        )
    if tokenizer.get_vocab_size(with_added_tokens=True) != spec.vocab_size:
        raise RuntimeError("tokenizer vocabulary does not match configured model vocabulary")
    return tokenizer


def save_tokenizer(tokenizer: Tokenizer, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(path), pretty=True)


def load_tokenizer(path: Path) -> Tokenizer:
    tokenizer = Tokenizer.from_file(str(path))
    missing = [token for token in SPECIAL_TOKENS if tokenizer.token_to_id(token) is None]
    if missing:
        raise ValueError(f"tokenizer lacks required special tokens: {missing}")
    return tokenizer


def reserved_token_ids(tokenizer: Tokenizer) -> frozenset[int]:
    """Return every deterministic padding-vocabulary ID for inference suppression."""
    return frozenset(
        token_id
        for token, token_id in tokenizer.get_vocab(with_added_tokens=True).items()
        if token.startswith(RESERVED_TOKEN_PREFIX) and token.endswith("|>")
    )


def _ids(tokenizer: Tokenizer, text: str) -> tuple[int, ...]:
    return tuple(tokenizer.encode(text, add_special_tokens=False).ids)


def _required_token_id(tokenizer: Tokenizer, token: str) -> int:
    token_id = tokenizer.token_to_id(token)
    if token_id is None:
        raise ValueError(f"tokenizer lacks required token {token}")
    return token_id


def encode_conditional_example(tokenizer: Tokenizer, example: ConditionalExample) -> TokenSequence:
    """Encode prompt and thought as context; poem tokens alone receive loss by default."""
    bos_id = _required_token_id(tokenizer, "<|bos|>")
    eos_id = _required_token_id(tokenizer, "<|eos|>")
    prompt_id = _required_token_id(tokenizer, "<|prompt|>")
    thought_id = _required_token_id(tokenizer, "<|thought|>")
    poem_id = _required_token_id(tokenizer, "<|poem|>")
    prefix: list[int] = [bos_id, prompt_id, *_ids(tokenizer, example.prompt)]
    if example.thought is not None:
        prefix.extend((thought_id, *_ids(tokenizer, example.thought)))
    prefix.extend((poem_id,))
    target = [*_ids(tokenizer, example.poem_target), eos_id]
    input_ids = tuple(prefix + target)
    if example.loss_on_poem_only:
        loss_mask = (False,) * len(prefix) + (True,) * len(target)
    else:
        loss_mask = (True,) * len(input_ids)
    return TokenSequence(
        example_id=example.example_id,
        boundary_key=example.leakage_key,
        input_ids=input_ids,
        loss_mask=loss_mask,
        objective="conditional_poetry",
    )


def encode_auxiliary_prose_ntp_example(
    tokenizer: Tokenizer, example: ProseNTPExample
) -> TokenSequence:
    """Encode the separately configured raw-prose next-token objective."""
    bos_id = _required_token_id(tokenizer, "<|bos|>")
    eos_id = _required_token_id(tokenizer, "<|eos|>")
    input_ids = (bos_id, *_ids(tokenizer, example.text), eos_id)
    return TokenSequence(
        example_id=example.example_id,
        boundary_key=example.document_id,
        input_ids=input_ids,
        loss_mask=(False,) + (True,) * (len(input_ids) - 1),
        objective="auxiliary_prose_ntp",
    )
