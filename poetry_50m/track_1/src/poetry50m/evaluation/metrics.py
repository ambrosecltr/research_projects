"""Transparent, deterministic quality and contamination measurements."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from poetry50m.data.schema import TokenSequence

WORD = re.compile(r"[\w']+", re.UNICODE)


def _words(text: str) -> tuple[str, ...]:
    return tuple(word.casefold() for word in WORD.findall(text))


def _ngrams(words: Sequence[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(words[index : index + n]) for index in range(max(0, len(words) - n + 1))}


@dataclass(frozen=True, slots=True)
class OverlapMetrics:
    exact_match: bool
    maximum_ngram: int
    ngram_overlap_rate: float
    per_ngram: tuple[NGramOverlap, ...]


@dataclass(frozen=True, slots=True)
class NGramOverlap:
    n: int
    candidate_count: int
    matching_count: int
    rate: float


def training_overlap(
    generated: str, training_texts: Iterable[str], *, max_n: int = 12
) -> OverlapMetrics:
    """Measure word-level training overlap without claiming a causal plagiarism verdict."""
    if max_n < 1:
        raise ValueError("max_n must be positive")
    generated_words = _words(generated)
    generated_normalised = " ".join(generated_words)
    training_documents = tuple(_words(text) for text in training_texts)
    exact = bool(generated_normalised) and any(
        generated_normalised == " ".join(document) for document in training_documents
    )
    maximum = 0
    per_ngram: list[NGramOverlap] = []
    for n in range(1, min(max_n, len(generated_words)) + 1):
        candidate = _ngrams(generated_words, n)
        training_ngrams = set().union(*(_ngrams(document, n) for document in training_documents))
        overlap = candidate & training_ngrams
        if overlap:
            maximum = n
        per_ngram.append(
            NGramOverlap(n, len(candidate), len(overlap), len(overlap) / max(1, len(candidate)))
        )
    longest = next((item for item in reversed(per_ngram) if item.matching_count), None)
    return OverlapMetrics(
        exact_match=exact,
        maximum_ngram=maximum,
        ngram_overlap_rate=longest.rate if longest else 0.0,
        per_ngram=tuple(per_ngram),
    )


@dataclass(frozen=True, slots=True)
class DegenerationMetrics:
    repeated_line_rate: float
    repeated_bigram_rate: float
    longest_repeated_phrase: int


def repetition_metrics(text: str) -> DegenerationMetrics:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    repeated_lines = len(lines) - len(set(lines))
    words = _words(text)
    bigrams = [tuple(words[index : index + 2]) for index in range(max(0, len(words) - 1))]
    counts = Counter(bigrams)
    repeated_bigrams = sum(count - 1 for count in counts.values() if count > 1)
    longest = 0
    for n in range(1, len(words) // 2 + 1):
        phrases = _ngrams(words, n)
        if len(phrases) < len(words) - n + 1:
            longest = n
    return DegenerationMetrics(
        repeated_line_rate=repeated_lines / max(1, len(lines)),
        repeated_bigram_rate=repeated_bigrams / max(1, len(bigrams)),
        longest_repeated_phrase=longest,
    )


@dataclass(frozen=True, slots=True)
class StructuralMetrics:
    line_count: int
    stanza_count: int
    mean_words_per_line: float
    line_length_variance: float


def structural_metrics(text: str) -> StructuralMetrics:
    stanzas = [stanza for stanza in re.split(r"\n[ \t]*\n", text.strip()) if stanza.strip()]
    lines = [line for stanza in stanzas for line in stanza.splitlines() if line.strip()]
    lengths = [len(_words(line)) for line in lines]
    mean = sum(lengths) / max(1, len(lengths))
    variance = sum((length - mean) ** 2 for length in lengths) / max(1, len(lengths))
    return StructuralMetrics(len(lines), len(stanzas), mean, variance)


def keyword_relevance(text: str, keywords: Iterable[str]) -> float:
    generated = set(_words(text))
    target = {word.casefold() for keyword in keywords for word in _words(keyword)}
    if not target:
        raise ValueError("keywords must contain at least one word")
    return len(generated & target) / len(target)


@dataclass(frozen=True, slots=True)
class HeldoutLossInput:
    """The evaluation boundary supplied to a trainer/loss implementation."""

    example_id: str
    input_ids: tuple[int, ...]
    target_positions: tuple[int, ...]


def heldout_loss_inputs(sequences: Iterable[TokenSequence]) -> tuple[HeldoutLossInput, ...]:
    values: list[HeldoutLossInput] = []
    for sequence in sequences:
        targets = tuple(index for index, enabled in enumerate(sequence.loss_mask) if enabled)
        if not targets:
            raise ValueError(f"held-out sequence {sequence.example_id} has no targets")
        values.append(HeldoutLossInput(sequence.example_id, sequence.input_ids, targets))
    return tuple(values)
