from __future__ import annotations

import json
from pathlib import Path

import pytest

from poetry50m.data.artifacts import (
    read_conditional_examples,
    read_packed_sequences,
    read_pairings,
    read_prompt_records,
    read_prose_examples,
    read_thought_records,
    write_pairings,
    write_prompt_records,
    write_thought_records,
)
from poetry50m.data.batch_stream import PreparedBatchStream
from poetry50m.data.loaders import write_manifest
from poetry50m.data.packing import PackedSequence
from poetry50m.data.prepare import (
    PreparedDataConfig,
    load_preparation_config,
    load_prepared_data,
    prepare_data,
)
from poetry50m.data.schema import (
    ContentBlock,
    CrossDocumentPairing,
    ObjectiveMix,
    PromptRecord,
    Provenance,
    SourceDocument,
    ThoughtRecord,
    TokenSequence,
)
from poetry50m.data.splits import SplitRatios, split_for_key
from poetry50m.data.tokenizer import TokenizerSpec, train_tokenizer


def document() -> SourceDocument:
    text = "Wild geese call.\n\nThe river waits."
    return SourceDocument(
        "poem-1",
        Provenance(
            "Fixture",
            "Fixture",
            "licensed",
            "fixture",
            rights_status="licensed",
            rights_evidence="fixture",
        ),
        text,
        (
            ContentBlock("poem-1:poem", "poem", text, poem_id="wild-geese", title="Wild Geese"),
            ContentBlock(
                "poem-1:stanza:0",
                "stanza",
                "Wild geese call.",
                poem_id="wild-geese",
                stanza_index=0,
            ),
            ContentBlock(
                "poem-1:stanza:1",
                "stanza",
                "The river waits.",
                poem_id="wild-geese",
                stanza_index=1,
            ),
            ContentBlock("poem-1:paragraph:0", "paragraph", text, paragraph_index=0),
        ),
    )


def prepare_fixture(tmp_path: Path, mix: ObjectiveMix | None = None) -> Path:
    mix = mix or ObjectiveMix(1.0, 0.0)
    corpus = tmp_path / "corpus.jsonl"
    prompts = tmp_path / "prompts.jsonl"
    thoughts = tmp_path / "thoughts.jsonl"
    write_manifest(corpus, [document()])
    write_prompt_records(
        prompts, [PromptRecord("title", "poem-1", "Write wild geese.", "title", "fixture")]
    )
    write_thought_records(
        thoughts, [ThoughtRecord("thought", "poem-1", "Attention waits.", "editorial", "fixture")]
    )
    output = tmp_path / "prepared"
    prepare_data(
        corpus_manifest=corpus,
        prompt_records=prompts,
        thought_records=thoughts,
        pairings=None,
        output_directory=output,
        config=PreparedDataConfig(
            SplitRatios(0.98, 0.01, 0.01),
            "fixture-salt",
            TokenizerSpec(vocab_size=300, min_frequency=1),
            10,
            mix,
        ),
    )
    return output


def test_canonical_records_and_prepared_artifact_round_trip_into_batch(tmp_path: Path):
    prompt_path = tmp_path / "prompts.jsonl"
    thought_path = tmp_path / "thoughts.jsonl"
    pairing_path = tmp_path / "pairings.jsonl"
    prompt = PromptRecord("prompt", "poem-1", "Write a river.", "imagery", "fixture")
    thought = ThoughtRecord("thought", "book-1", "Attention is rare.", "passage", "fixture")
    pairing = CrossDocumentPairing(
        "pair", "poem-1", "poem-1:poem", "prompt", "thought", ("fixture",)
    )
    write_prompt_records(prompt_path, [prompt])
    write_thought_records(thought_path, [thought])
    write_pairings(pairing_path, [pairing])
    assert read_prompt_records(prompt_path) == (prompt,)
    assert read_thought_records(thought_path) == (thought,)
    assert read_pairings(pairing_path) == (pairing,)
    prepared = prepare_fixture(tmp_path)
    loaded = load_prepared_data(prepared)
    packs = read_packed_sequences(loaded.train_packs_path)
    conditional = read_conditional_examples(prepared / "train.conditional.jsonl")
    assert conditional
    assert {example.thought is None for example in conditional} == {False, True}
    assert read_prose_examples(prepared / "train.prose.jsonl")
    stream = PreparedBatchStream(packs, batch_size=2, pad_token_id=0, seed=7)
    batch = next(stream)
    assert batch["input_ids"].shape == batch["targets"].shape
    assert isinstance(batch["data_token_count"], int) and batch["data_token_count"] > 0


def test_artifact_decoder_rejects_non_string_transformation_lineage(tmp_path: Path):
    path = tmp_path / "pairings.jsonl"
    path.write_text(
        json.dumps(
            {
                "pairing_id": "pair",
                "target_document_id": "poem-1",
                "target_block_id": "poem-1:poem",
                "prompt_id": "prompt",
                "thought_id": None,
                "transformation_lineage": ["editorial", 1],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-empty strings"):
        read_pairings(path)


def test_stream_resume_order_rejection_and_skip_accounting(tmp_path: Path):
    prepared = prepare_fixture(tmp_path)
    packs = read_packed_sequences(prepared / "train.packed.jsonl")
    first = PreparedBatchStream(packs, batch_size=1, pad_token_id=0, seed=3)
    advance = first.skip_batches(2)
    state = first.state_dict()
    resumed = PreparedBatchStream(packs, batch_size=1, pad_token_id=0, seed=3)
    resumed.load_state_dict(state)
    assert next(first)["example_ids"] == next(resumed)["example_ids"]
    assert advance.batch_count == 2 and advance.data_token_count > 0
    bad = dict(state)
    bad["order_digest"] = "not-the-order"
    with pytest.raises(ValueError, match="order mismatch"):
        resumed.load_state_dict(bad)


def test_strict_and_experimental_cyclic_curricula_have_stable_orders(tmp_path: Path):
    packs = read_packed_sequences(prepare_fixture(tmp_path) / "train.packed.jsonl")
    difficulty = {
        f"{pack.objective}:pack:{pack.pack_id}": float(index) for index, pack in enumerate(packs)
    }
    strict = PreparedBatchStream(
        packs,
        batch_size=1,
        pad_token_id=0,
        curriculum="strict_hard_to_easy",
        difficulty=difficulty,
    )
    cyclic = PreparedBatchStream(
        packs,
        batch_size=1,
        pad_token_id=0,
        curriculum="cyclic_hard_to_easy",
        difficulty=difficulty,
    )
    strict_first = next(strict)["example_ids"]
    for _ in range(len(packs)):
        next(cyclic)
    assert next(cyclic)["example_ids"] != strict_first or len(packs) == 1


def test_auxiliary_objective_weight_requires_and_changes_train_material(tmp_path: Path):
    poetry_only = prepare_fixture(tmp_path / "poetry", ObjectiveMix(1.0, 0.0))
    mixed = prepare_fixture(tmp_path / "mixed", ObjectiveMix(1.0, 1.0))
    poetry_packs = read_packed_sequences(poetry_only / "train.packed.jsonl")
    mixed_packs = read_packed_sequences(mixed / "train.packed.jsonl")
    assert len(mixed_packs) > len(poetry_packs)
    assert (
        load_prepared_data(mixed).metadata["config"]["objective_mix"]["auxiliary_prose_ntp"] == 1.0
    )


def test_prepared_metadata_reports_realized_supervised_token_mix(tmp_path: Path):
    prepared = prepare_fixture(tmp_path, ObjectiveMix(1.0, 0.25))
    artifact = load_prepared_data(prepared)
    packs = read_packed_sequences(prepared / "train.packed.jsonl")
    stats = artifact.metadata["train_objective_stats"]
    total_supervised = sum(sum(pack.loss_mask) for pack in packs)

    for objective in ("conditional_poetry", "auxiliary_prose_ntp"):
        objective_packs = tuple(pack for pack in packs if pack.objective == objective)
        supervised_tokens = sum(sum(pack.loss_mask) for pack in objective_packs)
        assert stats[objective]["pack_count"] == len(objective_packs)
        assert stats[objective]["supervised_token_count"] == supervised_tokens
        assert stats[objective]["supervised_token_ratio"] == pytest.approx(
            supervised_tokens / total_supervised
        )
    assert sum(values["supervised_token_ratio"] for values in stats.values()) == pytest.approx(1.0)
    assert artifact.metadata["config"]["objective_mix"] == {
        "conditional_poetry": 1.0,
        "auxiliary_prose_ntp": 0.25,
    }


def test_heldout_near_duplicate_prose_is_excluded_before_tokenizer_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    words = tuple(f"image{index:02d}" for index in range(60))
    heldout_poem = "\n".join(" ".join(words[index : index + 10]) for index in range(0, 60, 10))
    changed = list(words)
    changed[30] = "altered-image"
    leaked_prose = (
        "A short editorial preface surrounds this quoted passage. "
        + ", ".join(changed)
        + ". A short editorial afterword follows the quoted passage."
    )
    unrelated_prose = "Attention moves quietly through an unrelated field of ordinary prose."

    def source(document_id: str, text: str, blocks: tuple[ContentBlock, ...]) -> SourceDocument:
        return SourceDocument(
            document_id,
            Provenance(
                "Fixture",
                "Fixture",
                "licensed",
                "fixture",
                rights_status="licensed",
                rights_evidence="fixture",
            ),
            text,
            blocks,
        )

    target = source(
        "heldout-poem",
        heldout_poem,
        (
            ContentBlock("heldout-poem:poem", "poem", heldout_poem, poem_id="heldout-family"),
            ContentBlock("heldout-poem:paragraph", "paragraph", heldout_poem, paragraph_index=0),
        ),
    )
    leaked = source(
        "copied-prose",
        leaked_prose,
        (ContentBlock("copied-prose:paragraph", "paragraph", leaked_prose, paragraph_index=0),),
    )
    unrelated = source(
        "unrelated-prose",
        unrelated_prose,
        (
            ContentBlock(
                "unrelated-prose:paragraph", "paragraph", unrelated_prose, paragraph_index=0
            ),
        ),
    )
    ratios = SplitRatios(0.34, 0.33, 0.33)
    target_key = " ".join(words)
    salt = next(
        f"near-duplicate-{index}"
        for index in range(10_000)
        if split_for_key(target_key, ratios, salt=f"near-duplicate-{index}") != "train"
        and split_for_key("copied-prose", ratios, salt=f"near-duplicate-{index}") == "train"
        and split_for_key("unrelated-prose", ratios, salt=f"near-duplicate-{index}") == "train"
    )
    corpus = tmp_path / "corpus.jsonl"
    prompts = tmp_path / "prompts.jsonl"
    thoughts = tmp_path / "thoughts.jsonl"
    write_manifest(corpus, (target, leaked, unrelated))
    write_prompt_records(
        prompts,
        (
            PromptRecord(
                "heldout-prompt", target.document_id, "Write the images.", "theme", "fixture"
            ),
        ),
    )
    write_thought_records(thoughts, ())
    tokenizer_inputs: list[str] = []

    def capture_tokenizer_inputs(texts: list[str], spec: TokenizerSpec):
        captured = tuple(texts)
        tokenizer_inputs.extend(captured)
        return train_tokenizer(captured, spec)

    monkeypatch.setattr("poetry50m.data.prepare.train_tokenizer", capture_tokenizer_inputs)
    output = tmp_path / "prepared"
    artifact = prepare_data(
        corpus_manifest=corpus,
        prompt_records=prompts,
        thought_records=thoughts,
        pairings=None,
        output_directory=output,
        config=PreparedDataConfig(
            ratios,
            salt,
            TokenizerSpec(vocab_size=300, min_frequency=1),
            32,
            ObjectiveMix(1.0, 1.0),
        ),
    )

    prose = read_prose_examples(output / "train.prose.jsonl")
    assert [example.block_id for example in prose] == ["unrelated-prose:paragraph"]
    assert unrelated_prose in tokenizer_inputs
    assert leaked_prose not in tokenizer_inputs
    assert heldout_poem not in tokenizer_inputs
    copied_exclusion = next(
        item
        for item in artifact.metadata["excluded_prose"]
        if item["block_id"] == "copied-prose:paragraph"
    )
    assert copied_exclusion["reason"] == "heldout_lexical_family"
    assert copied_exclusion["evidence"]["metric"] == "shingle_containment"
    assert copied_exclusion["evidence"]["score"] >= copied_exclusion["evidence"]["threshold"]
    assert copied_exclusion["evidence"]["heldout_field"] == "poem_target"


def test_objective_scheduler_tracks_weights_and_rejects_changed_pack_contents():
    conditional = tuple(
        PackedSequence(index, "poem", (f"poetry-{index}",), (1, 2, 3), (False, True, True))
        for index in range(3)
    )
    prose = (
        PackedSequence(
            0,
            "prose",
            ("prose-0",),
            (4, 5, 6),
            (False, True, True),
            "auxiliary_prose_ntp",
        ),
    )
    groups = {"conditional_poetry": conditional, "auxiliary_prose_ntp": prose}
    stream = PreparedBatchStream(
        groups, batch_size=1, pad_token_id=0, objective_mix=ObjectiveMix(1.0, 0.1)
    )
    observed = [next(stream)["example_ids"][0].split(":pack:")[0] for _ in range(22)]
    assert observed.count("conditional_poetry") == 20
    assert observed.count("auxiliary_prose_ntp") == 2
    thirds = PreparedBatchStream(
        groups, batch_size=1, pad_token_id=0, objective_mix=ObjectiveMix(1.0, 0.3)
    )
    third_observed = [next(thirds)["example_ids"][0].split(":pack:")[0] for _ in range(13)]
    assert third_observed.count("conditional_poetry") == 10
    assert third_observed.count("auxiliary_prose_ntp") == 3
    state = stream.state_dict()
    resumed = PreparedBatchStream(
        groups, batch_size=1, pad_token_id=0, objective_mix=ObjectiveMix(1.0, 0.1)
    )
    resumed.load_state_dict(state)
    assert next(stream)["example_ids"] == next(resumed)["example_ids"]
    batch = next(resumed)
    assert len(batch["example_ids"]) == batch["input_ids"].shape[0]
    changed = (
        PackedSequence(0, "poem", ("poetry-0",), (1, 9, 3), (False, True, True)),
        *conditional[1:],
    )
    with pytest.raises(ValueError, match="invalid prepared-stream state"):
        PreparedBatchStream(
            {"conditional_poetry": changed, "auxiliary_prose_ntp": prose},
            batch_size=1,
            pad_token_id=0,
            objective_mix=ObjectiveMix(1.0, 0.1),
        ).load_state_dict(state)


def test_objective_scheduler_preserves_small_decimal_ratios_and_resume_boundaries():
    conditional = (PackedSequence(0, "poem", ("poetry",), (1, 2, 3), (False, True, True)),)
    prose = (
        PackedSequence(
            0,
            "prose",
            ("prose",),
            (4, 5, 6),
            (False, True, True),
            "auxiliary_prose_ntp",
        ),
    )
    groups = {"conditional_poetry": conditional, "auxiliary_prose_ntp": prose}
    thousand_to_one = PreparedBatchStream(
        groups,
        batch_size=1,
        pad_token_id=0,
        objective_mix=ObjectiveMix(1.0, 0.001),
    )
    observed = [next(thousand_to_one)["example_ids"][0].split(":pack:")[0] for _ in range(2_002)]
    assert observed.count("conditional_poetry") == 2_000
    assert observed.count("auxiliary_prose_ntp") == 2

    one_to_two = PreparedBatchStream(
        groups,
        batch_size=1,
        pad_token_id=0,
        objective_mix=ObjectiveMix(0.001, 0.002),
    )
    observed = [next(one_to_two)["example_ids"][0].split(":pack:")[0] for _ in range(6)]
    assert observed.count("conditional_poetry") == 2
    assert observed.count("auxiliary_prose_ntp") == 4

    original = PreparedBatchStream(
        groups,
        batch_size=1,
        pad_token_id=0,
        objective_mix=ObjectiveMix(1.0, 0.001),
    )
    original.skip_batches(1_000)
    resumed = PreparedBatchStream(
        groups,
        batch_size=1,
        pad_token_id=0,
        objective_mix=ObjectiveMix(1.0, 0.001),
    )
    resumed.load_state_dict(original.state_dict())
    assert [next(original)["example_ids"] for _ in range(4)] == [
        next(resumed)["example_ids"] for _ in range(4)
    ]


def test_positive_missing_objective_and_boolean_token_ids_are_rejected():
    conditional = (PackedSequence(0, "poem", ("poetry",), (1, 2), (False, True)),)
    prose = (PackedSequence(0, "prose", ("prose",), (3, 4), (False, True), "auxiliary_prose_ntp"),)
    with pytest.raises(ValueError, match="auxiliary_prose_ntp"):
        PreparedBatchStream(
            {"conditional_poetry": conditional},
            batch_size=1,
            pad_token_id=0,
            objective_mix=ObjectiveMix(1.0, 1.0),
        )
    with pytest.raises(ValueError, match="conditional_poetry"):
        PreparedBatchStream(
            {"auxiliary_prose_ntp": prose},
            batch_size=1,
            pad_token_id=0,
            objective_mix=ObjectiveMix(1.0, 1.0),
        )
    with pytest.raises(ValueError, match="non-negative"):
        TokenSequence("bad", "bad", (True,), (True,))
    with pytest.raises(ValueError, match="pack_id"):
        PackedSequence(True, "poem", ("example",), (1, 2), (False, True))
    with pytest.raises(ValueError, match="token IDs"):
        PackedSequence(0, "poem", ("example",), (1, True), (False, True))
    with pytest.raises(ValueError, match="loss mask"):
        PackedSequence(0, "poem", ("example",), (1, 2), (False, 1))
    stream = PreparedBatchStream(conditional, batch_size=1, pad_token_id=0)
    with pytest.raises(ValueError, match="skip count"):
        stream.skip_batches(True)
    with pytest.raises(ValueError, match="difficulty"):
        PreparedBatchStream(
            conditional,
            batch_size=1,
            pad_token_id=0,
            curriculum="strict_hard_to_easy",
            difficulty={"conditional_poetry:pack:0": float("nan")},
        )


def test_nested_data_config_unknown_keys_are_rejected(tmp_path: Path):
    config = {
        "format_version": 1,
        "manifest_format": "jsonl",
        "manifest_schema": "SourceDocument",
        "split": {"salt": "x", "train": 0.9, "validation": 0.05, "test": 0.05},
        "tokenizer": {"vocab_size": 300, "min_frequency": 1, "special_tokens": []},
        "packing": {"sequence_length": 8, "typo": True},
        "objectives": {"conditional_poetry": 1.0, "auxiliary_prose_ntp": 0.0},
        "rights": {"allow_synthetic": False},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown packing"):
        load_preparation_config(path)


def test_data_config_rejects_boolean_format_version(tmp_path: Path):
    config = {
        "format_version": True,
        "manifest_format": "jsonl",
        "manifest_schema": "SourceDocument",
        "split": {"salt": "x", "train": 0.9, "validation": 0.05, "test": 0.05},
        "tokenizer": {"vocab_size": 300, "min_frequency": 1, "special_tokens": []},
        "packing": {"sequence_length": 8},
        "objectives": {"conditional_poetry": 1.0, "auxiliary_prose_ntp": 0.0},
        "rights": {"allow_synthetic": False},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported"):
        load_preparation_config(path)


def test_preparation_rejects_nonempty_output(tmp_path: Path):
    output = tmp_path / "prepared"
    output.mkdir()
    (output / "stale.txt").write_text("stale", encoding="utf-8")
    with pytest.raises(FileExistsError, match="not empty"):
        prepare_data(
            corpus_manifest=tmp_path / "missing-corpus.jsonl",
            prompt_records=tmp_path / "missing-prompts.jsonl",
            thought_records=tmp_path / "missing-thoughts.jsonl",
            pairings=None,
            output_directory=output,
            config=PreparedDataConfig(
                SplitRatios(0.9, 0.05, 0.05),
                "fixture",
                TokenizerSpec(vocab_size=300),
                8,
                ObjectiveMix(),
            ),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("split_salt", ""),
        ("sequence_length", True),
        ("allow_synthetic", 1),
    ],
)
def test_prepared_config_rejects_invalid_boundary_types(field: str, value: object):
    arguments: dict[str, object] = {
        "split_ratios": SplitRatios(),
        "split_salt": "salt",
        "tokenizer": TokenizerSpec(vocab_size=300),
        "sequence_length": 8,
        "objective_mix": ObjectiveMix(),
        "allow_synthetic": False,
    }
    arguments[field] = value
    with pytest.raises(ValueError):
        PreparedDataConfig(**arguments)


@pytest.mark.parametrize("field", ["conditional_poetry", "auxiliary_prose_ntp"])
@pytest.mark.parametrize("value", [True, "1", float("nan"), float("inf")])
def test_objective_mix_rejects_invalid_numbers(field: str, value: object):
    arguments: dict[str, object] = {
        "conditional_poetry": 1.0,
        "auxiliary_prose_ntp": 1.0,
    }
    arguments[field] = value
    with pytest.raises(ValueError):
        ObjectiveMix(**arguments)
