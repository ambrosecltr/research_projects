from __future__ import annotations

import json
from typing import cast

import pytest

from poetry50m.data.artifacts import read_prompt_records, write_prompt_records
from poetry50m.data.difficulty import DifficultyLedger, DifficultyRecord
from poetry50m.data.examples import (
    build_auxiliary_prose_ntp_examples,
    build_conditional_examples,
    build_poetry_ntp_examples,
)
from poetry50m.data.loaders import assert_ingestible, iter_manifest, write_manifest
from poetry50m.data.packing import chunk_sequence, pack_sequences
from poetry50m.data.schema import (
    ConditionalExample,
    ContentBlock,
    CrossDocumentPairing,
    ObjectiveMix,
    PromptRecord,
    Provenance,
    SourceDocument,
    ThoughtRecord,
    TokenSequence,
)
from poetry50m.data.splits import (
    SplitRatios,
    assert_no_poem_leakage,
    split_examples,
    split_for_key,
)
from poetry50m.data.tokenizer import (
    RESERVED_TOKEN_PREFIX,
    SPECIAL_TOKENS,
    TokenizerSpec,
    encode_auxiliary_prose_ntp_example,
    encode_conditional_example,
    encode_poetry_ntp_example,
    load_tokenizer,
    reserved_token_ids,
    save_tokenizer,
    train_tokenizer,
)


def source_document() -> SourceDocument:
    return SourceDocument(
        document_id="devotions-001",
        provenance=Provenance(
            "Devotions", "Mary Oliver", "synthetic", "fixture", rights_status="synthetic"
        ),
        text="Wild geese\ncall.\n\nThe river waits.",
        blocks=(
            ContentBlock(
                "devotions-001:poem",
                "poem",
                "Wild geese\ncall.\n\nThe river waits.",
                poem_id="wild-geese",
                title="Wild Geese",
            ),
            ContentBlock(
                "devotions-001:stanza:0",
                "stanza",
                "Wild geese\ncall.",
                poem_id="wild-geese",
                stanza_index=0,
            ),
            ContentBlock(
                "devotions-001:stanza:1",
                "stanza",
                "The river waits.",
                poem_id="wild-geese",
                stanza_index=1,
            ),
        ),
    )


def test_manifest_round_trip_preserves_provenance_and_boundaries(tmp_path):
    path = tmp_path / "corpus.jsonl"
    write_manifest(path, [source_document()], allow_synthetic=True)
    restored = tuple(iter_manifest(path, allow_synthetic=True))
    assert restored == (source_document(),)
    assert restored[0].blocks[1].text == "Wild geese\ncall."
    assert restored[0].raw_content_hash and restored[0].cleaned_content_hash


def test_gutenberg_book_verse_has_an_explicit_unconditional_objective():
    document = SourceDocument(
        document_id="gutenberg:123",
        provenance=Provenance(
            "Public domain verse",
            "Anonymous",
            "public_domain",
            "fixture",
            rights_status="public_domain",
            rights_evidence="fixture public-domain record",
        ),
        text="A first line\nA second line",
        blocks=(
            ContentBlock("gutenberg:123:verse", "verse_document", "A first line\nA second line"),
        ),
        metadata={"training_role": "unconditional_book_verse_ntp"},
    )
    examples = build_poetry_ntp_examples((document,))
    assert len(examples) == 1
    tokenizer = train_tokenizer((document.text,), TokenizerSpec(vocab_size=512))
    sequence = encode_poetry_ntp_example(tokenizer, examples[0])
    assert sequence.objective == "poetry_ntp"
    assert sequence.boundary_key == document.document_id
    assert all(sequence.loss_mask[1:])


def test_manifest_schema_rejects_nested_coercions_and_weak_types(tmp_path):
    payload = source_document().to_mapping()
    payload["blocks"][0]["metadata"] = {"source_page": 1}
    path = tmp_path / "bad-metadata.jsonl"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="metadata"):
        tuple(iter_manifest(path, allow_synthetic=True))

    payload = source_document().to_mapping()
    payload["transformation_lineage"] = ["clean", 1]
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="transformation_lineage"):
        tuple(iter_manifest(path, allow_synthetic=True))

    with pytest.raises(TypeError, match="loss_on_poem_only"):
        ConditionalExample(
            "conditional",
            "document",
            "poem",
            "prompt",
            "target",
            loss_on_poem_only=cast(bool, 1),
        )


def test_public_jsonl_readers_reject_blank_records_with_line_evidence(tmp_path):
    manifest_path = tmp_path / "corpus.jsonl"
    write_manifest(manifest_path, [source_document()], allow_synthetic=True)
    manifest_path.write_text(manifest_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"blank manifest record .*:2"):
        tuple(iter_manifest(manifest_path, allow_synthetic=True))

    prompt_path = tmp_path / "prompts.jsonl"
    write_prompt_records(
        prompt_path,
        [PromptRecord("prompt", "document", "Write a river.", "imagery", "fixture")],
    )
    prompt_path.write_text(prompt_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"blank artifact record .*:2"):
        read_prompt_records(prompt_path)

    difficulty_path = tmp_path / "difficulty.jsonl"
    DifficultyLedger([DifficultyRecord("example", 1, 1.0, 0)]).save(difficulty_path)
    difficulty_path.write_text(difficulty_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"blank difficulty record .*:2"):
        DifficultyLedger.load(difficulty_path)


def test_rights_gate_preserves_unknown_and_rejects_explicit_denial(tmp_path):
    unreviewed = SourceDocument(
        "unknown-1",
        Provenance("Unknown", "Unknown", "not_asserted", "fixture"),
        "Unreviewed text.",
        (
            ContentBlock(
                "unknown-1:paragraph:0", "paragraph", "Unreviewed text.", paragraph_index=0
            ),
        ),
    )
    assert_ingestible(unreviewed)
    write_manifest(tmp_path / "unknown.jsonl", [unreviewed])
    denied = SourceDocument(
        "denied-1",
        Provenance("Denied", "Denied", "not_asserted", "fixture", rights_status="denied"),
        "Denied text.",
        (ContentBlock("denied-1:paragraph:0", "paragraph", "Denied text.", paragraph_index=0),),
    )
    try:
        write_manifest(tmp_path / "blocked.jsonl", [denied])
    except PermissionError:
        pass
    else:
        raise AssertionError("manifest writer accepted explicitly denied rights")


def test_conditional_examples_and_poem_family_split_cannot_leak():
    prompt = PromptRecord("prompt-1", "devotions-001", "A flock returns", "imagery", "fixture")
    thought = ThoughtRecord(
        "thought-1", "devotions-001", "Attention is prayer.", "editorial", "fixture"
    )
    examples = build_conditional_examples([source_document()], prompts=[prompt], thoughts=[thought])
    assert len(examples) == 6
    for target in {example.poem_target for example in examples}:
        variants = [example for example in examples if example.poem_target == target]
        assert {example.thought for example in variants} == {None, "Attention is prayer."}
    assert all(example.prompt_id == "prompt-1" for example in examples)
    assert all(
        example.thought_id == ("thought-1" if example.thought is not None else None)
        for example in examples
    )
    same_poem = ConditionalExample("other", "other-document", "wild-geese", "River", "Other poem")
    splits = split_examples((*examples, same_poem), SplitRatios(0.34, 0.33, 0.33), salt="fixture")
    assert_no_poem_leakage(splits)
    owner = next(
        name
        for name, items in splits.items()
        if any(item.example_id == examples[0].example_id for item in items)
    )
    assert any(item.example_id == "other" for item in splits[owner])


def test_source_document_and_conditioning_document_bind_split_family():
    first = ConditionalExample(
        "first",
        "poem-doc",
        "poem-a",
        "Prompt A",
        "Target A",
        prompt_id="prompt-a",
        prompt_document_id="philosophy-doc",
    )
    second = ConditionalExample(
        "second",
        "poem-doc",
        "poem-b",
        "Prompt B",
        "Different target",
        prompt_id="prompt-b",
        prompt_document_id="other-philosophy-doc",
    )
    third = ConditionalExample(
        "third",
        "another-poem-doc",
        "poem-c",
        "Prompt C",
        "Third target",
        prompt_id="prompt-c",
        prompt_document_id="philosophy-doc",
    )
    splits = split_examples((first, second, third), SplitRatios(0.34, 0.33, 0.33), salt="docs")
    owners = {example.example_id: name for name, examples in splits.items() for example in examples}
    assert owners["first"] == owners["second"] == owners["third"]


def test_cross_field_near_duplicates_bind_conditional_examples_to_one_split():
    words = tuple(f"line{index:02d}" for index in range(60))
    heldout_target = " ".join(words)
    copied = list(words)
    copied[30] = "altered-line"
    copied_prompt = "Preface words " + "\n".join(copied) + " closing words"
    unrelated_target = "A wholly separate target about a bell beneath winter branches"
    first = ConditionalExample(
        "heldout",
        "heldout-document",
        "heldout-poem",
        "An unrelated short prompt.",
        heldout_target,
        prompt_id="heldout-prompt",
        prompt_document_id="heldout-document",
    )
    second = ConditionalExample(
        "copied",
        "copied-document",
        "copied-poem",
        copied_prompt,
        unrelated_target,
        prompt_id="copied-prompt",
        prompt_document_id="copied-document",
    )
    ratios = SplitRatios(0.34, 0.33, 0.33)
    salt = next(
        f"cross-field-{index}"
        for index in range(10_000)
        if split_for_key(heldout_target, ratios, salt=f"cross-field-{index}")
        != split_for_key(unrelated_target.casefold(), ratios, salt=f"cross-field-{index}")
    )
    splits = split_examples((first, second), ratios, salt=salt)
    owners = {example.example_id: name for name, examples in splits.items() for example in examples}
    assert owners["heldout"] == owners["copied"]
    assert_no_poem_leakage(splits)


def test_reused_boilerplate_prompt_does_not_collapse_distinct_targets():
    first = ConditionalExample(
        "first",
        "first-document",
        "first-poem",
        "Write a poem in this author's style.",
        "a river carries moonlight beyond the reeds",
    )
    second = ConditionalExample(
        "second",
        "second-document",
        "second-poem",
        "Write a poem in this author's style.",
        "an iron bell wakes sparrows beneath winter roofs",
    )
    ratios = SplitRatios(0.34, 0.33, 0.33)
    salt = next(
        f"shared-prompt-{index}"
        for index in range(10_000)
        if split_for_key(first.poem_target, ratios, salt=f"shared-prompt-{index}")
        != split_for_key(second.poem_target, ratios, salt=f"shared-prompt-{index}")
    )

    splits = split_examples((first, second), ratios, salt=salt)
    owners = {example.example_id: name for name, examples in splits.items() for example in examples}

    assert owners["first"] != owners["second"]
    assert_no_poem_leakage(splits)


def test_cross_document_pairing_keeps_philosophy_thought_and_poem_target_distinct():
    poem = source_document()
    philosophy = SourceDocument(
        "weil-001",
        Provenance(
            "Gravity and Grace", "Simone Weil", "synthetic", "fixture", rights_status="synthetic"
        ),
        "Attention is the rarest generosity.",
        (
            ContentBlock(
                "weil-001:paragraph:0",
                "paragraph",
                "Attention is the rarest generosity.",
                paragraph_index=0,
            ),
        ),
    )
    prompt = PromptRecord(
        "theme", "weil-001", "Write about attention beside a river.", "theme", "fixture"
    )
    ordinary_prompt = PromptRecord(
        "title", "devotions-001", "Write about wild geese.", "title", "fixture"
    )
    thought = ThoughtRecord(
        "weil-thought", "weil-001", "Attention is the rarest generosity.", "passage", "fixture"
    )
    pairing = CrossDocumentPairing(
        "pair-1",
        "devotions-001",
        "devotions-001:stanza:0",
        "theme",
        "weil-thought",
        ("editorial_pairing",),
    )
    examples = build_conditional_examples(
        [poem, philosophy],
        prompts=[prompt, ordinary_prompt],
        thoughts=[thought],
        pairings=[pairing],
    )
    paired_examples = [example for example in examples if example.pairing_id == "pair-1"]
    assert len(paired_examples) == 2
    paired = next(example for example in paired_examples if example.thought is not None)
    prompt_only = next(example for example in paired_examples if example.thought is None)
    ordinary = next(example for example in examples if example.pairing_id is None)
    assert paired.document_id == "devotions-001"
    assert paired.thought == thought.text and paired.poem_target == "Wild geese\ncall."
    assert paired.prompt_id == "theme" and paired.thought_id == "weil-thought"
    assert paired.prompt_document_id == "weil-001"
    assert paired.thought_document_id == "weil-001"
    assert paired.transformation_lineage == ("editorial_pairing",)
    assert prompt_only.prompt == prompt.prompt
    assert prompt_only.poem_target == paired.poem_target
    assert prompt_only.prompt_document_id == "weil-001"
    assert prompt_only.thought_id is None and prompt_only.thought_document_id is None
    assert prompt_only.transformation_lineage == ("editorial_pairing",)
    assert ordinary.prompt_id is not None
    prose = build_auxiliary_prose_ntp_examples([philosophy])
    assert len(prose) == 1
    tokenizer = train_tokenizer([philosophy.text], TokenizerSpec(vocab_size=300, min_frequency=1))
    assert all(encode_auxiliary_prose_ntp_example(tokenizer, prose[0]).loss_mask[1:])
    assert ObjectiveMix(conditional_poetry=1.0, auxiliary_prose_ntp=0.0).auxiliary_prose_ntp == 0.0


def test_explicit_local_pairing_wins_identity_collision_without_duplicate_sequences():
    document = source_document()
    prompt = PromptRecord(
        "local-prompt", document.document_id, "Write about returning geese.", "theme", "fixture"
    )
    thought = ThoughtRecord(
        "local-thought", document.document_id, "Attention follows return.", "editorial", "fixture"
    )
    pairing = CrossDocumentPairing(
        "local-pairing",
        document.document_id,
        "devotions-001:stanza:0",
        prompt.prompt_id,
        thought.thought_id,
        ("explicit_local_relation",),
    )
    examples = build_conditional_examples(
        (document,), prompts=(prompt,), thoughts=(thought,), pairings=(pairing,)
    )
    target_examples = [
        example for example in examples if example.poem_target == "Wild geese\ncall."
    ]
    assert len(target_examples) == 2
    assert {example.thought for example in target_examples} == {
        None,
        "Attention follows return.",
    }
    assert all(example.pairing_id == pairing.pairing_id for example in target_examples)
    assert all(
        example.transformation_lineage == pairing.transformation_lineage
        for example in target_examples
    )
    assert len(
        {(example.prompt, example.thought, example.poem_target) for example in examples}
    ) == len(examples)


def test_byte_fallback_tokenizer_round_trip_and_target_only_loss(tmp_path):
    tokenizer = train_tokenizer(
        ["Wild geese\ncall.", "A river ��� waits."], TokenizerSpec(vocab_size=300, min_frequency=1)
    )
    assert tokenizer.decode(tokenizer.encode("Wild geese\ncall.").ids) == "Wild geese\ncall."
    example = ConditionalExample(
        "one", "doc", "poem", "wild geese", "A river\nwaits.", thought="attention"
    )
    encoded = encode_conditional_example(tokenizer, example)
    first_target = encoded.loss_mask.index(True)
    assert not any(encoded.loss_mask[:first_target])
    assert all(encoded.loss_mask[first_target:])
    path = tmp_path / "tokenizer.json"
    save_tokenizer(tokenizer, path)
    reloaded = load_tokenizer(path)
    assert reloaded.get_vocab() == tokenizer.get_vocab()
    assert tokenizer.get_vocab_size(with_added_tokens=True) == 300
    expected_reserved = frozenset(
        token_id
        for token, token_id in tokenizer.get_vocab(with_added_tokens=True).items()
        if token.startswith(RESERVED_TOKEN_PREFIX) and token.endswith("|>")
    )
    assert reserved_token_ids(tokenizer) == expected_reserved
    assert expected_reserved
    try:
        TokenizerSpec(vocab_size=262)
    except ValueError:
        pass
    else:
        raise AssertionError("byte alphabet shortfall was accepted")
    try:
        TokenizerSpec(vocab_size=300, special_tokens=tuple(reversed(SPECIAL_TOKENS)))
    except ValueError:
        pass
    else:
        raise AssertionError("noncanonical special-token contract was accepted")


def test_packing_is_lossless_and_never_crosses_a_poem_family():
    sequences = (
        TokenSequence("a", "poem-a", (1, 2), (False, True)),
        TokenSequence("b", "poem-a", (3,), (True,)),
        TokenSequence("c", "poem-b", (4, 5), (False, True)),
    )
    packs = pack_sequences(sequences, sequence_length=3)
    assert [pack.input_ids for pack in packs] == [(1, 2, 3), (4, 5)]
    assert [pack.boundary_key for pack in packs] == ["poem-a", "poem-b"]
    long = TokenSequence("long", "poem-c", (1, 2, 3, 4, 5), (False, True, True, True, True))
    long_packs = pack_sequences([long], sequence_length=3)
    assert sum(sum(pack.loss_mask) for pack in long_packs) == 4
    long_prefix = TokenSequence("prefix", "poem-d", tuple(range(12)), (False,) * 9 + (True,) * 3)
    chunks = chunk_sequence(long_prefix, sequence_length=4)
    assert all(any(chunk.loss_mask) for chunk in chunks)
    assert sum(sum(chunk.loss_mask) for chunk in chunks) == 3


def test_difficulty_ledger_is_deterministic_and_persistent(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    ledger = DifficultyLedger(
        [
            DifficultyRecord("easy", 2, 1.0, 0),
            DifficultyRecord("hard", 2, 6.0, 0),
            DifficultyRecord("middle", 2, 3.0, 0),
        ]
    )
    ledger_path = tmp_path / "difficulty.jsonl"
    ledger.save(ledger_path)
    assert DifficultyLedger.load(ledger_path).for_pass() == ledger.for_pass()
    original = ledger_path.read_bytes()

    def fail_replace(source, destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr("poetry50m.data.difficulty.os.replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        ledger.save(ledger_path)
    assert ledger_path.read_bytes() == original
    assert not tuple(tmp_path.glob(".difficulty.jsonl.*.tmp"))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("token_count", True),
        ("negative_log_likelihood", False),
        ("negative_log_likelihood", float("nan")),
        ("pass_index", True),
    ],
)
def test_difficulty_record_rejects_boolean_and_nonfinite_fields(field: str, value: object):
    arguments: dict[str, object] = {
        "example_id": "example",
        "token_count": 1,
        "negative_log_likelihood": 1.0,
        "pass_index": 0,
    }
    arguments[field] = value
    with pytest.raises(ValueError, match="difficulty"):
        DifficultyRecord(**arguments)
