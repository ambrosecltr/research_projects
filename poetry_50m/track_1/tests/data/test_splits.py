from __future__ import annotations

import random
from collections.abc import Iterable

import pytest

import poetry50m.data.splits as splits_module
from poetry50m.data.schema import ConditionalExample, SplitName
from poetry50m.data.splits import (
    LexicalFamilyHit,
    LexicalFamilyIndex,
    LexicalFamilyMatch,
    assert_no_poem_leakage,
    lexical_family_match,
    split_examples,
)


def _components(example_count: int, edges: Iterable[tuple[int, int]]) -> frozenset[frozenset[int]]:
    parents = list(range(example_count))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left, right in edges:
        union(left, right)
    members: dict[int, set[int]] = {}
    for index in range(example_count):
        members.setdefault(find(index), set()).add(index)
    return frozenset(frozenset(group) for group in members.values())


def _pairwise_family_edges(
    examples: tuple[ConditionalExample, ...],
) -> Iterable[tuple[int, int]]:
    for right, example in enumerate(examples):
        for left, other in enumerate(examples[:right]):
            same_poem = example.poem_id is not None and example.poem_id == other.poem_id
            shared_document = bool(example.source_document_ids & other.source_document_ids)
            shared_prompt = example.prompt_id is not None and example.prompt_id == other.prompt_id
            shared_thought = (
                example.thought_id is not None and example.thought_id == other.thought_id
            )
            lexical_match = any(
                lexical_family_match(left_text, right_text) is not None
                for left_text in (
                    other.prompt,
                    *((other.thought,) if other.thought is not None else ()),
                    other.poem_target,
                )
                for right_text in (
                    example.prompt,
                    *((example.thought,) if example.thought is not None else ()),
                    example.poem_target,
                )
            )
            if same_poem or shared_document or shared_prompt or shared_thought or lexical_match:
                yield left, right


def _unique_example(
    index: int,
    *,
    poem_id: str | None = None,
    prompt: str | None = None,
    poem_target: str | None = None,
    thought: str | None = None,
    prompt_id: str | None = None,
    prompt_document_id: str | None = None,
    thought_id: str | None = None,
    thought_document_id: str | None = None,
    split: SplitName | None = None,
) -> ConditionalExample:
    return ConditionalExample(
        example_id=f"example-{index:04d}",
        document_id=f"document-{index}",
        poem_id=poem_id or f"poem-{index}",
        prompt=prompt or f"compose distinct prompt token_{index}",
        poem_target=poem_target or " ".join(f"target_{index}_{word}" for word in range(12)),
        thought=thought,
        prompt_id=prompt_id,
        prompt_document_id=prompt_document_id,
        thought_id=thought_id,
        thought_document_id=thought_document_id,
        split=split,
    )


def test_lexical_family_index_matches_pairwise_oracle_in_reference_order() -> None:
    containment_words = tuple(f"containment_{index}" for index in range(20))
    jaccard_words = tuple(f"jaccard_{index}" for index in range(60))
    jaccard_copy = list(jaccard_words)
    jaccard_copy[30] = "altered"
    weak_prefix = ("common", "opening", "phrase", "shared", "widely")
    references = (
        ("...", "empty-first"),
        ("Exact, phrase!", "exact-first"),
        (" ".join(containment_words), "containment"),
        ("!!!", "empty-second"),
        ("exact phrase", "exact-second"),
        (" ".join(jaccard_copy), "jaccard"),
        (" ".join((*weak_prefix, *(f"reference_{index}" for index in range(20)))), "weak"),
        ("entirely unrelated reference text", "unrelated"),
    )
    queries = (
        "???",
        "EXACT phrase",
        " ".join((*containment_words, *(f"extension_{index}" for index in range(20)))),
        " ".join(jaccard_words),
        " ".join((*weak_prefix, *(f"query_{index}" for index in range(20)))),
        "no reference resembles this query",
    )
    index = LexicalFamilyIndex(references)

    for query in queries:
        expected = tuple(
            LexicalFamilyHit(match, payload)
            for reference, payload in references
            if (match := lexical_family_match(query, reference)) is not None
        )
        assert index.find_matches(query) == expected


def test_lexical_family_index_matches_exhaustive_pairwise_oracle() -> None:
    texts = tuple(
        " ".join(
            f"{'left' if bitmask & (1 << word_index) else 'right'}_{word_index}"
            for word_index in range(10)
        )
        for bitmask in range(1 << 7)
    )
    references = tuple((text, index) for index, text in enumerate(texts))
    index = LexicalFamilyIndex(references)

    for query in texts:
        expected = tuple(
            LexicalFamilyHit(match, payload)
            for reference, payload in references
            if (match := lexical_family_match(query, reference)) is not None
        )
        assert index.find_matches(query) == expected


def test_lexical_family_index_matches_oracle_across_prefix_boundary_sizes() -> None:
    references = tuple(
        (
            " ".join(f"size_{size}_{index}" for index in range(size + 4)),
            f"size-{size}",
        )
        for size in (63, 64, 65)
    )
    queries: list[str] = []
    for reference, _ in references:
        near_copy = reference.split()
        near_copy[len(near_copy) // 2] = "changed_boundary_word"
        queries.extend((" ".join(near_copy), f"{reference} extension words for containment"))
    index = LexicalFamilyIndex(references)

    for query in queries:
        expected = tuple(
            LexicalFamilyHit(match, payload)
            for reference, payload in references
            if (match := lexical_family_match(query, reference)) is not None
        )
        assert index.find_matches(query) == expected


def test_lexical_family_index_matches_randomized_pairwise_oracle() -> None:
    generator = random.Random(20_260_726)
    vocabulary = tuple(f"word_{index}" for index in range(30))
    reference_tokens = tuple(
        tuple(generator.choice(vocabulary) for _ in range(generator.randrange(36)))
        for _ in range(240)
    )
    references = tuple(
        (
            " ".join(tokens) if tokens else generator.choice(("...", "!!!", "???")),
            f"reference-{index}",
        )
        for index, tokens in enumerate(reference_tokens)
    )
    queries = ["???", "entirely unseen query words"]
    for query_index in range(160):
        tokens = list(generator.choice(reference_tokens))
        operation = query_index % 5
        if operation == 0:
            query = " ".join(tokens).upper()
        elif operation == 1 and tokens:
            tokens[generator.randrange(len(tokens))] = f"changed_{query_index}"
            query = " ".join(tokens)
        elif operation == 2:
            tokens.extend(f"extension_{query_index}_{index}" for index in range(12))
            query = " ".join(tokens)
        elif operation == 3 and len(tokens) > 1:
            query = " ".join(tokens[: generator.randint(1, len(tokens) - 1)])
        else:
            query = " ".join(generator.choice(vocabulary) for _ in range(generator.randrange(36)))
        queries.append(query or "...")
    index = LexicalFamilyIndex(references)

    for query in queries:
        expected = tuple(
            LexicalFamilyHit(match, payload)
            for reference, payload in references
            if (match := lexical_family_match(query, reference)) is not None
        )
        assert index.find_matches(query) == expected
    expected_pairs = tuple(
        (left_id, right_id, match)
        for right_id, right in enumerate(index._signatures)
        for left_id, left in enumerate(index._signatures[:right_id])
        if (
            match := lexical_family_match(
                right.normalized_text,
                left.normalized_text,
            )
        )
        is not None
    )
    assert tuple(index._matching_signature_pairs()) == expected_pairs


def test_title_prompt_prefix_filter_avoids_common_posting_quadratic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signature_count = 8_263
    index = LexicalFamilyIndex(
        (
            f"Write a poem titled The unique_{signature_id}",
            signature_id,
        )
        for signature_id in range(signature_count)
    )
    assert max(map(len, index._postings.values())) == signature_count
    assert max(map(len, index._prefix_postings.values())) == 1
    original_posting_ids = splits_module._posting_ids_before
    posting_iterations = 0

    def counted_posting_ids(
        posting: tuple[int, ...],
        upper_signature_id: int | None,
    ) -> Iterable[int]:
        nonlocal posting_iterations
        for signature_id in original_posting_ids(posting, upper_signature_id):
            posting_iterations += 1
            if posting_iterations > signature_count * 2:
                pytest.fail("prefix filtering traversed a common title posting")
            yield signature_id

    monkeypatch.setattr(splits_module, "_posting_ids_before", counted_posting_ids)

    assert tuple(index._matching_signature_pairs()) == ()
    assert posting_iterations == 0


def test_indexed_family_components_match_pairwise_semantics() -> None:
    shared_words = tuple(f"shared_{index}" for index in range(60))
    near_copy = list(shared_words)
    near_copy[30] = "altered"
    contained_words = tuple(f"contained_{index}" for index in range(20))
    containing_words = (*contained_words, *(f"extension_{index}" for index in range(20)))
    weak_overlap_left = tuple(f"weak_{index}" for index in range(10))
    weak_overlap_right = (*weak_overlap_left[:5], *(f"different_{index}" for index in range(5)))
    examples = (
        _unique_example(0, poem_id="shared-poem"),
        _unique_example(1, poem_id="shared-poem"),
        _unique_example(
            2,
            prompt_id="prompt-2",
            prompt_document_id="shared-conditioning-document",
        ),
        _unique_example(
            3,
            prompt_id="prompt-3",
            prompt_document_id="shared-conditioning-document",
        ),
        _unique_example(4, prompt_id="shared-prompt", prompt_document_id="prompt-document-4"),
        _unique_example(5, prompt_id="shared-prompt", prompt_document_id="prompt-document-5"),
        _unique_example(
            6,
            thought="distinct thought six",
            thought_id="shared-thought",
            thought_document_id="thought-document-6",
        ),
        _unique_example(
            7,
            thought="distinct thought seven",
            thought_id="shared-thought",
            thought_document_id="thought-document-7",
        ),
        _unique_example(8, poem_target="Moon, river!"),
        _unique_example(9, prompt="moon river"),
        _unique_example(10, poem_target=" ".join(shared_words)),
        _unique_example(11, prompt=" ".join(near_copy)),
        _unique_example(12),
        _unique_example(13, poem_target=" ".join(contained_words)),
        _unique_example(14, prompt=" ".join(containing_words)),
        _unique_example(15, poem_target=" ".join(weak_overlap_left)),
        _unique_example(16, prompt=" ".join(weak_overlap_right)),
        _unique_example(17, poem_id="cross-namespace-id"),
        _unique_example(
            18,
            prompt_id="cross-namespace-id",
            prompt_document_id="prompt-document-18",
        ),
    )

    indexed = _components(len(examples), splits_module._indexed_family_edges(examples))
    pairwise = _components(len(examples), _pairwise_family_edges(examples))

    assert indexed == pairwise
    assert frozenset({13, 14}) in indexed
    assert frozenset({15}) in indexed
    assert frozenset({16}) in indexed
    assert frozenset({17}) in indexed
    assert frozenset({18}) in indexed


def test_assert_no_poem_leakage_detects_indexed_cross_split_match() -> None:
    words = tuple(f"line_{index}" for index in range(60))
    near_copy = list(words)
    near_copy[30] = "altered"
    train = _unique_example(0, poem_target=" ".join(words), split="train")
    test = _unique_example(1, prompt=" ".join(near_copy), split="test")

    with pytest.raises(ValueError, match="leakage family spans train and test"):
        assert_no_poem_leakage({"train": (train,), "validation": (), "test": (test,)})


def test_thousands_of_unrelated_examples_avoid_pairwise_lexical_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared_prefix = ("common", "opening", "phrase", "shared", "widely")
    examples = tuple(
        _unique_example(
            index,
            poem_target=" ".join(
                (*shared_prefix, *(f"unique_{index}_{word}" for word in range(20)))
            ),
        )
        for index in range(3_000)
    )
    original_match = splits_module.lexical_family_match
    match_calls = 0

    def bounded_match(
        left_text: str,
        right_text: str,
        *,
        left_shingles: frozenset[tuple[str, ...]] | None = None,
        right_shingles: frozenset[tuple[str, ...]] | None = None,
    ) -> LexicalFamilyMatch | None:
        nonlocal match_calls
        match_calls += 1
        if match_calls > 100:
            pytest.fail("lexical matching regressed to scanning unrelated example pairs")
        return original_match(
            left_text,
            right_text,
            left_shingles=left_shingles,
            right_shingles=right_shingles,
        )

    monkeypatch.setattr(splits_module, "lexical_family_match", bounded_match)
    lexical_index = LexicalFamilyIndex(
        (example.poem_target, example.example_id) for example in examples
    )
    query = " ".join((*shared_prefix, *(f"query_unique_{word}" for word in range(20))))
    assert lexical_index.find_matches(query) == ()
    result = split_examples(examples)
    assert sum(len(items) for items in result.values()) == len(examples)
    assert_no_poem_leakage(result)
    assert match_calls == 0
