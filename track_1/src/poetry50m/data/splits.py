"""Stable content-hash splits with poem-family leakage prevention."""

from __future__ import annotations

import math
import re
from bisect import bisect_left
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Generic, Literal, TypeVar

from .schema import ConditionalExample, SplitName

LEXICAL_FAMILY_THRESHOLD = 0.8
_Shingle = tuple[str, ...]
_ShingleSet = frozenset[_Shingle]
_Payload = TypeVar("_Payload")


@dataclass(frozen=True, slots=True)
class LexicalFamilyMatch:
    metric: Literal["normalized_exact", "shingle_jaccard", "shingle_containment"]
    score: float
    shared_shingles: int
    comparison_shingles: int


@dataclass(frozen=True, slots=True)
class LexicalFamilyHit(Generic[_Payload]):
    match: LexicalFamilyMatch
    payload: _Payload


@dataclass(frozen=True, slots=True)
class _IndexedSignature(Generic[_Payload]):
    normalized_text: str
    shingle_ids: _ShingleSet
    ordered_shingle_ids: tuple[_Shingle, ...]
    entries: tuple[tuple[int, _Payload], ...]


@dataclass(frozen=True, slots=True)
class SplitRatios:
    train: float = 0.9
    validation: float = 0.05
    test: float = 0.05

    def __post_init__(self) -> None:
        values = (self.train, self.validation, self.test)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in values
        ):
            raise ValueError("split ratios must be finite numbers")
        if any(value <= 0 for value in values) or abs(sum(values) - 1.0) > 1e-9:
            raise ValueError("split ratios must be positive and sum to 1")


def split_for_key(
    key: str, ratios: SplitRatios | None = None, *, salt: str = "poetry50m-v1"
) -> SplitName:
    """Assign every occurrence of an identical poem/document family to one split."""
    if not key:
        raise ValueError("split key cannot be empty")
    ratios = ratios or SplitRatios()
    fraction = int.from_bytes(sha256(f"{salt}\0{key}".encode()).digest()[:8], "big") / 2**64
    if fraction < ratios.train:
        return "train"
    if fraction < ratios.train + ratios.validation:
        return "validation"
    return "test"


def split_examples(
    examples: Iterable[ConditionalExample],
    ratios: SplitRatios | None = None,
    *,
    salt: str = "poetry50m-v1",
) -> dict[SplitName, tuple[ConditionalExample, ...]]:
    """Assign examples by poem ID when present, never separately by stanza/example."""
    ratios = ratios or SplitRatios()
    ordered = tuple(sorted(examples, key=lambda item: item.example_id))
    parents = list(range(len(ordered)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        larger_root, smaller_root = max(left_root, right_root), min(left_root, right_root)
        parents[smaller_root] = larger_root

    for left, right in _indexed_family_edges(ordered):
        union(left, right)
    family_members: dict[int, list[ConditionalExample]] = {}
    for index, example in enumerate(ordered):
        family_members.setdefault(find(index), []).append(example)
    buckets: dict[SplitName, list[ConditionalExample]] = {"train": [], "validation": [], "test": []}
    for members in family_members.values():
        family_key = "\0".join(sorted(_normalised_target(item.poem_target) for item in members))
        split = split_for_key(family_key, ratios, salt=salt)
        for example in members:
            buckets[split].append(
                ConditionalExample(
                    example_id=example.example_id,
                    document_id=example.document_id,
                    poem_id=example.poem_id,
                    prompt=example.prompt,
                    thought=example.thought,
                    poem_target=example.poem_target,
                    prompt_id=example.prompt_id,
                    thought_id=example.thought_id,
                    prompt_document_id=example.prompt_document_id,
                    thought_document_id=example.thought_document_id,
                    pairing_id=example.pairing_id,
                    transformation_lineage=example.transformation_lineage,
                    loss_on_poem_only=example.loss_on_poem_only,
                    split=split,
                )
            )
    return {name: tuple(items) for name, items in buckets.items()}


def _normalised_target(text: str) -> str:
    return " ".join(re.findall(r"[\w']+", text.casefold()))


def _normalized_shingles(normalized_text: str, width: int = 5) -> _ShingleSet:
    words = normalized_text.split()
    if len(words) < width:
        return frozenset({tuple(words)})
    return frozenset(tuple(words[index : index + width]) for index in range(len(words) - width + 1))


def _shingles(text: str, width: int = 5) -> _ShingleSet:
    return _normalized_shingles(_normalised_target(text), width)


def _source_texts(example: ConditionalExample) -> tuple[str, ...]:
    """Texts whose reuse would expose held-out target or source material."""
    return (
        *((example.thought,) if example.thought is not None else ()),
        example.poem_target,
    )


def lexical_family_match(
    left_text: str,
    right_text: str,
    *,
    left_shingles: _ShingleSet | None = None,
    right_shingles: _ShingleSet | None = None,
) -> LexicalFamilyMatch | None:
    """Return deterministic lexical evidence only for an exact or strong family match."""
    left_normalized = _normalised_target(left_text)
    right_normalized = _normalised_target(right_text)
    if left_normalized == right_normalized:
        shingles = left_shingles if left_shingles is not None else _shingles(left_text)
        return LexicalFamilyMatch("normalized_exact", 1.0, len(shingles), len(shingles))
    left = left_shingles if left_shingles is not None else _shingles(left_text)
    right = right_shingles if right_shingles is not None else _shingles(right_text)
    shared = len(left & right)
    union = len(left | right)
    jaccard = shared / union if union else 1.0
    if jaccard >= LEXICAL_FAMILY_THRESHOLD:
        return LexicalFamilyMatch("shingle_jaccard", jaccard, shared, union)
    smaller = min(len(left), len(right))
    containment = shared / smaller if smaller else 1.0
    if containment >= LEXICAL_FAMILY_THRESHOLD:
        return LexicalFamilyMatch("shingle_containment", containment, shared, smaller)
    return None


class LexicalFamilyIndex(Generic[_Payload]):
    """Index fixed reference texts for exact lexical-family queries.

    References shorter than a query contribute only their global-rarity prefix
    to a compact posting.  Longer references use the global posting and the
    query prefix.  Any overlap of ``ceil(0.8 * min(size))`` has an element in
    the respective prefixes, so this filter is complete for both supported
    similarity metrics.
    """

    __slots__ = (
        "_postings",
        "_prefix_postings",
        "_reference_sizes",
        "_signatures",
    )

    _postings: dict[_Shingle, tuple[int, ...]]
    _prefix_postings: dict[_Shingle, tuple[int, ...]]
    _reference_sizes: tuple[int, ...]
    _signatures: tuple[_IndexedSignature[_Payload], ...]

    def __init__(self, entries: Iterable[tuple[str, _Payload]]) -> None:
        signature_ids: dict[str, int] = {}
        normalized_texts: list[str] = []
        signature_shingle_ids: list[_ShingleSet] = []
        signature_entries: list[list[tuple[int, _Payload]]] = []
        for position, (text, payload) in enumerate(entries):
            normalized_text = _normalised_target(text)
            signature_id = signature_ids.get(normalized_text)
            if signature_id is None:
                signature_id = len(normalized_texts)
                signature_ids[normalized_text] = signature_id
                normalized_texts.append(normalized_text)
                signature_shingle_ids.append(_normalized_shingles(normalized_text))
                signature_entries.append([])
            signature_entries[signature_id].append((position, payload))

        mutable_postings: dict[_Shingle, list[int]] = {}
        for signature_id, shingle_ids_for_signature in enumerate(signature_shingle_ids):
            for shingle in shingle_ids_for_signature:
                mutable_postings.setdefault(shingle, []).append(signature_id)
        self._postings = {shingle: tuple(posting) for shingle, posting in mutable_postings.items()}
        self._reference_sizes = tuple(len(ids) for ids in signature_shingle_ids)
        self._signatures = tuple(
            _IndexedSignature(
                normalized_text=normalized_text,
                shingle_ids=shingles,
                ordered_shingle_ids=(
                    tuple(
                        sorted(
                            shingles, key=lambda shingle: (len(self._postings[shingle]), shingle)
                        )
                    )
                    if len(shingles) <= 64
                    else ()
                ),
                entries=tuple(signature_entries[signature_id]),
            )
            for signature_id, (normalized_text, shingles) in enumerate(
                zip(normalized_texts, signature_shingle_ids, strict=True)
            )
        )
        mutable_prefix_postings: dict[_Shingle, list[int]] = {}
        for signature_id, signature in enumerate(self._signatures):
            if len(signature.shingle_ids) > 64:
                continue
            for shingle in signature.ordered_shingle_ids[
                : _prefix_length(len(signature.shingle_ids))
            ]:
                mutable_prefix_postings.setdefault(shingle, []).append(signature_id)
        self._prefix_postings = {
            shingle: tuple(posting) for shingle, posting in mutable_prefix_postings.items()
        }

    def find_matches(self, text: str) -> tuple[LexicalFamilyHit[_Payload], ...]:
        """Return every matching payload in its original reference order."""
        normalized_text = _normalised_target(text)
        query_shingles = _normalized_shingles(normalized_text)
        positioned_hits = [
            (position, LexicalFamilyHit(match, payload))
            for signature_id, match in self._matching_signatures(
                normalized_text,
                query_shingles,
            )
            for position, payload in self._signatures[signature_id].entries
        ]
        positioned_hits.sort(key=lambda item: item[0])
        return tuple(hit for _, hit in positioned_hits)

    def _matching_signature_pairs(
        self,
    ) -> Iterator[tuple[int, int, LexicalFamilyMatch]]:
        for right_id, right in enumerate(self._signatures):
            for left_id, match in self._matching_signatures(
                right.normalized_text,
                right.shingle_ids,
                upper_signature_id=right_id,
            ):
                yield left_id, right_id, match

    def _matching_signatures(
        self,
        normalized_text: str,
        query_shingles: _ShingleSet,
        *,
        upper_signature_id: int | None = None,
    ) -> Iterator[tuple[int, LexicalFamilyMatch]]:
        query_size = len(query_shingles)
        candidates: Iterable[tuple[int, int]]
        if query_size > 64:
            shared_counts: dict[int, int] = {}
            for shingle in query_shingles:
                for signature_id in _posting_ids_before(
                    self._postings.get(shingle, ()), upper_signature_id
                ):
                    shared_counts[signature_id] = shared_counts.get(signature_id, 0) + 1
            candidates = shared_counts.items()
        else:
            ordered = tuple(
                sorted(
                    query_shingles,
                    key=lambda shingle: (len(self._postings.get(shingle, ())), shingle),
                )
            )
            candidate_ids: set[int] = set()
            for position, shingle in enumerate(ordered):
                if position < _prefix_length(query_size):
                    for signature_id in _posting_ids_before(
                        self._postings.get(shingle, ()), upper_signature_id
                    ):
                        if self._reference_sizes[signature_id] >= query_size:
                            candidate_ids.add(signature_id)
                for signature_id in _posting_ids_before(
                    self._prefix_postings.get(shingle, ()), upper_signature_id
                ):
                    reference_size = self._reference_sizes[signature_id]
                    if (
                        reference_size < query_size
                        and position < query_size - _required_overlap(reference_size) + 1
                    ):
                        candidate_ids.add(signature_id)
            candidates = (
                (signature_id, len(self._signatures[signature_id].shingle_ids & query_shingles))
                for signature_id in candidate_ids
            )
        for signature_id, shared in sorted(candidates):
            signature = self._signatures[signature_id]
            smaller = min(len(signature.shingle_ids), query_size)
            if shared < _required_overlap(smaller):
                continue
            match = lexical_family_match(normalized_text, signature.normalized_text)
            if match is not None:
                yield signature_id, match


def _required_overlap(size: int) -> int:
    return (4 * size + 4) // 5


def _prefix_length(size: int) -> int:
    return size - _required_overlap(size) + 1


def _posting_ids_before(posting: tuple[int, ...], upper_signature_id: int | None) -> Iterator[int]:
    """Yield posting IDs below an optional pairwise upper bound."""
    if upper_signature_id is None:
        yield from posting
        return
    yield from posting[: bisect_left(posting, upper_signature_id)]


def _identity_family_edges(
    examples: Sequence[ConditionalExample],
) -> Iterator[tuple[int, int]]:
    poem_owners: dict[str, int] = {}
    document_owners: dict[str, int] = {}
    prompt_owners: dict[str, int] = {}
    thought_owners: dict[str, int] = {}

    for index, example in enumerate(examples):
        connected_indices: set[int] = set()
        if example.poem_id is not None:
            prior = poem_owners.setdefault(example.poem_id, index)
            if prior != index:
                connected_indices.add(prior)
        for document_id in sorted(example.source_document_ids):
            prior = document_owners.setdefault(document_id, index)
            if prior != index:
                connected_indices.add(prior)
        if example.prompt_id is not None:
            prior = prompt_owners.setdefault(example.prompt_id, index)
            if prior != index:
                connected_indices.add(prior)
        if example.thought_id is not None:
            prior = thought_owners.setdefault(example.thought_id, index)
            if prior != index:
                connected_indices.add(prior)
        for connected_index in sorted(connected_indices):
            yield connected_index, index


def _lexical_family_edges(
    examples: Sequence[ConditionalExample],
) -> Iterator[tuple[int, int]]:
    source_index = LexicalFamilyIndex(
        (text, example_index)
        for example_index, example in enumerate(examples)
        for text in _source_texts(example)
    )
    for signature in source_index._signatures:
        representative = signature.entries[0][1]
        for _, example_index in signature.entries[1:]:
            yield representative, example_index
    for left_id, right_id, _ in source_index._matching_signature_pairs():
        yield (
            source_index._signatures[left_id].entries[0][1],
            source_index._signatures[right_id].entries[0][1],
        )
    # A prompt that copies source/target text must stay with that source
    # family. Boilerplate prompt-to-prompt similarity is not target leakage
    # and must not transitively collapse an otherwise valid held-out split.
    for example_index, example in enumerate(examples):
        for hit in source_index.find_matches(example.prompt):
            if hit.payload != example_index:
                yield hit.payload, example_index


def _indexed_family_edges(
    examples: Sequence[ConditionalExample],
) -> Iterator[tuple[int, int]]:
    yield from _identity_family_edges(examples)
    yield from _lexical_family_edges(examples)


def assert_no_poem_leakage(splits: Mapping[SplitName, Iterable[ConditionalExample]]) -> None:
    flattened = tuple(
        (split, example) for split, examples in splits.items() for example in examples
    )
    owners: dict[str, SplitName] = {}
    for split, example in flattened:
        prior = owners.setdefault(example.leakage_key, split)
        if prior != split:
            raise ValueError(f"leakage key {example.leakage_key!r} appears in {prior} and {split}")
    conditional_examples = tuple(example for _, example in flattened)
    for left, right in _indexed_family_edges(conditional_examples):
        left_split, right_split = flattened[left][0], flattened[right][0]
        if left_split != right_split:
            raise ValueError(f"leakage family spans {left_split} and {right_split}")
