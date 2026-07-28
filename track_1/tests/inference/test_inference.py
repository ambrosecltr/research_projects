from __future__ import annotations

import math
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer

from poetry50m.data import reserved_token_ids
from poetry50m.data.schema import ConditionalExample
from poetry50m.data.tokenizer import TokenizerSpec, encode_conditional_example, train_tokenizer
from poetry50m.evaluation.schema import (
    GenerationRequest,
    PromptCase,
    PromptSuite,
    generation_requests,
)
from poetry50m.inference import (
    GenerationConfig,
    GenerationRecord,
    GenerationResult,
    build_conditioning_tokens,
    generate,
    load_generation_records,
    load_snapshot_into_model,
    run_generation_manifest,
)
from poetry50m.inference.generation import _sample_top_p
from poetry50m.model import DecoderOnlyTransformer, ModelConfig
from poetry50m.training import mapping_hash
from poetry50m.trajectory.snapshots import save_weight_snapshot
from poetry50m.trajectory.types import SnapshotMetadata, WeightSnapshot


def tokenizer_and_model() -> tuple[Tokenizer, DecoderOnlyTransformer]:
    tokenizer = train_tokenizer(
        ["river stone\n", "moon over water\n", "attention becomes a bird"],
        TokenizerSpec(vocab_size=300, min_frequency=1),
    )
    model = DecoderOnlyTransformer(
        ModelConfig(
            architecture="gpt",
            vocab_size=tokenizer.get_vocab_size(),
            max_seq_len=32,
            d_model=16,
            n_layers=1,
            n_heads=4,
            ffn_dim=32,
        )
    )
    return tokenizer, model


def metadata(model_config: ModelConfig) -> SnapshotMetadata:
    return SnapshotMetadata(
        run_id="run-1",
        checkpoint_id="step-00000001",
        step=1,
        initialization_id="init-1",
        data_order_id="order-1",
        architecture_signature="tiny-gpt",
        corpus_signature="corpus-1",
        model_config_hash=mapping_hash(asdict(model_config)),
        tokenizer_hash="tokenizer-1",
        code_signature="code-1",
        training_config_hash="training-1",
    )


def generation_config(seed: int = 7) -> GenerationConfig:
    return GenerationConfig(max_new_tokens=4, temperature=0.8, top_p=0.9, seed=seed)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("max_new_tokens", True),
        ("seed", False),
        ("temperature", math.nan),
        ("temperature", math.inf),
        ("top_p", math.nan),
        ("top_p", True),
    ),
)
def test_generation_config_rejects_boolean_and_nonfinite_values(field: str, value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        replace(generation_config(), **{field: value})


def test_generation_result_and_record_reject_malformed_numeric_values() -> None:
    with pytest.raises(ValueError, match="integer token IDs"):
        GenerationResult((True,), (), "", "max_new_tokens", 0.0)
    with pytest.raises(ValueError, match="finite"):
        GenerationResult((1,), (), "", "max_new_tokens", math.nan)
    with pytest.raises(TypeError, match="seed"):
        GenerationRecord("request", "case", "step", True, (), "", "eos", 0, 0.0)
    with pytest.raises(TypeError, match="generated_token_count"):
        GenerationRecord("request", "case", "step", 1, (), "", "eos", False, 0.0)
    with pytest.raises(ValueError, match="non-negative integers"):
        GenerationRecord("request", "case", "step", 1, (True,), "", "eos", 1, 0.0)
    with pytest.raises(ValueError, match="finite"):
        GenerationRecord("request", "case", "step", 1, (), "", "eos", 0, math.inf)
    malformed = GenerationRecord("request", "case", "step", 1, (), "", "eos", 0, 0.0)
    value = malformed.to_mapping()
    value["generated_token_ids"] = [True]
    with pytest.raises(ValueError, match="integer list"):
        GenerationRecord.from_mapping(value)
    valid = malformed.to_mapping()
    valid["untracked"] = "not allowed"
    with pytest.raises(ValueError, match="exactly"):
        GenerationRecord.from_mapping(valid)


def test_conditioning_and_same_seed_generation_are_deterministic() -> None:
    tokenizer, model = tokenizer_and_model()
    model.train()
    first = generate(model, tokenizer, "river", generation_config(), thought="attention")
    second = generate(model, tokenizer, "river", generation_config(), thought="attention")
    assert model.training
    assert first.generated_token_ids == second.generated_token_ids
    assert first.generated_text == second.generated_text
    assert math.isfinite(first.wall_seconds) and first.wall_seconds >= 0.0
    prefix = build_conditioning_tokens(tokenizer, "river", "attention")
    assert first.conditioning_token_ids == prefix
    assert tokenizer.token_to_id("<|thought|>") in prefix
    prompt_only = build_conditioning_tokens(tokenizer, "river")
    assert tokenizer.token_to_id("<|thought|>") not in prompt_only
    training_sequence = encode_conditional_example(
        tokenizer,
        ConditionalExample("example", "document", "poem", "river", "stone and water"),
    )
    first_target = training_sequence.loss_mask.index(True)
    assert training_sequence.input_ids[:first_target] == prompt_only
    thought_training_sequence = encode_conditional_example(
        tokenizer,
        ConditionalExample(
            "thought-example",
            "document",
            "poem",
            "river",
            "stone and water",
            thought="attention",
        ),
    )
    first_thought_target = thought_training_sequence.loss_mask.index(True)
    assert thought_training_sequence.input_ids[:first_thought_target] == prefix


def _uncached_generated_ids(
    model: DecoderOnlyTransformer,
    tokenizer: Tokenizer,
    prompt: str,
    config: GenerationConfig,
) -> tuple[int, ...]:
    conditioning = build_conditioning_tokens(tokenizer, prompt)
    eos_id = tokenizer.token_to_id("<|eos|>")
    assert eos_id is not None
    structural_ids = {
        token_id
        for token in ("<|pad|>", "<|bos|>", "<|prompt|>", "<|thought|>", "<|poem|>", "<|mask|>")
        if (token_id := tokenizer.token_to_id(token)) is not None
    }
    suppressed = sorted(structural_ids.union(reserved_token_ids(tokenizer)))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)
    token_ids = list(conditioning)
    generated: list[int] = []
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for _ in range(config.max_new_tokens):
                logits = model(torch.tensor([token_ids], dtype=torch.long)).logits[0, -1].clone()
                logits[suppressed] = -torch.inf
                token_id = _sample_top_p(logits, config, generator)
                if token_id == eos_id:
                    break
                generated.append(token_id)
                token_ids.append(token_id)
    finally:
        model.train(was_training)
    return tuple(generated)


def test_cached_sampling_matches_uncached_reference() -> None:
    tokenizer, model = tokenizer_and_model()
    config = GenerationConfig(max_new_tokens=5, temperature=0.83, top_p=0.72, seed=113)
    expected = _uncached_generated_ids(model, tokenizer, "river", config)
    actual = generate(model, tokenizer, "river", config)
    assert actual.generated_token_ids == expected


def test_generation_restores_training_state_when_cached_forward_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer, model = tokenizer_and_model()
    model.train()

    def fail_cached_forward(*_: object, **__: object) -> object:
        raise RuntimeError("injected cached-forward failure")

    monkeypatch.setattr(model, "forward_cached", fail_cached_forward)
    with pytest.raises(RuntimeError, match="cached-forward failure"):
        generate(model, tokenizer, "river", generation_config())
    assert model.training


def test_generation_requires_exact_tokenizer_model_vocabulary_identity() -> None:
    tokenizer, model = tokenizer_and_model()
    mismatched = DecoderOnlyTransformer(
        replace(model.config, vocab_size=model.config.vocab_size + 1)
    )
    with pytest.raises(ValueError, match="vocabulary size must exactly match"):
        generate(mismatched, tokenizer, "river", generation_config())


def test_generation_rejects_out_of_range_reserved_token_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer, model = tokenizer_and_model()
    monkeypatch.setattr(
        "poetry50m.inference.generation.reserved_token_ids",
        lambda _: frozenset({model.config.vocab_size}),
    )
    with pytest.raises(ValueError, match="reserved tokenizer IDs"):
        generate(model, tokenizer, "river", generation_config())


def test_special_tokens_are_suppressed_and_context_is_enforced() -> None:
    tokenizer, model = tokenizer_and_model()
    structural_ids = [
        tokenizer.token_to_id(token)
        for token in ("<|pad|>", "<|bos|>", "<|prompt|>", "<|thought|>", "<|poem|>", "<|mask|>")
    ]
    poem_id = tokenizer.token_to_id("<|poem|>")
    padded_ids = reserved_token_ids(tokenizer)
    ordinary_id = tokenizer.encode("river", add_special_tokens=False).ids[0]
    assert padded_ids
    assert poem_id is not None and all(token_id is not None for token_id in structural_ids)
    assert ordinary_id not in padded_ids and ordinary_id not in structural_ids
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        embedding = model.token_embedding.weight
        dict(model.named_parameters())["final_norm.weight"].fill_(1.0)
        for token_id in structural_ids:
            assert token_id is not None
            embedding[token_id].fill_(10.0)
        for token_id in padded_ids:
            embedding[token_id].fill_(11.0)
        embedding[ordinary_id].fill_(9.0)
    result = generate(
        model,
        tokenizer,
        "river",
        GenerationConfig(max_new_tokens=2, temperature=1.0, top_p=1e-6, seed=1),
    )
    assert result.generated_token_ids == (ordinary_id, ordinary_id)
    assert not set(result.generated_token_ids).intersection(padded_ids)
    assert not set(result.generated_token_ids).intersection(
        int(token_id) for token_id in structural_ids
    )
    too_long = GenerationConfig(
        max_new_tokens=model.config.max_seq_len, temperature=1.0, top_p=1.0, seed=1
    )
    with pytest.raises(ValueError, match="exceeds the model context"):
        generate(model, tokenizer, "river", too_long)


def test_generation_stops_on_eos_with_cached_decode() -> None:
    tokenizer, model = tokenizer_and_model()
    eos_id = tokenizer.token_to_id("<|eos|>")
    poem_id = tokenizer.token_to_id("<|poem|>")
    assert eos_id is not None and poem_id is not None
    structural_ids = [
        tokenizer.token_to_id(token)
        for token in ("<|pad|>", "<|bos|>", "<|prompt|>", "<|thought|>", "<|poem|>", "<|mask|>")
    ]
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        embedding = model.token_embedding.weight
        dict(model.named_parameters())["final_norm.weight"].fill_(1.0)
        for token_id in structural_ids:
            assert token_id is not None
            embedding[token_id].fill_(30.0)
        for token_id in reserved_token_ids(tokenizer):
            embedding[token_id].fill_(30.0)
        embedding[eos_id].fill_(20.0)
    result = generate(
        model,
        tokenizer,
        "river",
        GenerationConfig(max_new_tokens=3, temperature=1.0, top_p=1e-6, seed=1),
    )
    assert result.generated_token_ids == ()
    assert result.stop_reason == "eos"


def test_snapshot_loading_is_restricted_and_coordinate_strict(tmp_path: Path) -> None:
    tokenizer, model = tokenizer_and_model()
    snapshot_path = tmp_path / "weights.pt"
    expected_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    expected_metadata = metadata(model.config)
    save_weight_snapshot(snapshot_path, WeightSnapshot(expected_metadata, expected_state))
    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    loaded = load_snapshot_into_model(model, snapshot_path, expected_metadata=expected_metadata)
    assert loaded == expected_metadata
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, expected_state[name], rtol=0, atol=0)


def test_snapshot_loading_rejects_metadata_for_a_different_model_config(tmp_path: Path) -> None:
    _, model = tokenizer_and_model()
    state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    snapshot_path = tmp_path / "wrong-model-config.pt"
    save_weight_snapshot(
        snapshot_path,
        WeightSnapshot(
            replace(metadata(model.config), model_config_hash="wrong-config"),
            state,
        ),
    )
    with pytest.raises(ValueError, match="configuration"):
        load_snapshot_into_model(model, snapshot_path)


def test_generation_manifest_execution_and_jsonl_roundtrip(tmp_path: Path) -> None:
    tokenizer, model = tokenizer_and_model()
    suite = PromptSuite(
        suite_id="small-suite",
        version=1,
        cases=(PromptCase("river", "river", ("river",)), PromptCase("moon", "moon", ("moon",))),
    )
    requests = generation_requests(
        suite,
        checkpoint_id="step-1",
        seed=5,
        max_new_tokens=3,
        temperature=0.9,
        top_p=0.95,
    )
    output_path = tmp_path / "generation.jsonl"
    records = run_generation_manifest(
        model,
        tokenizer,
        requests,
        output_path,
        thoughts_by_case={"river": "attention"},
    )
    assert {record.case_id for record in records} == {"river", "moon"}
    assert all(record.checkpoint_id == "step-1" for record in records)
    assert load_generation_records(output_path) == records


def test_manifest_rejects_unknown_thought_case(tmp_path: Path) -> None:
    tokenizer, model = tokenizer_and_model()
    request = GenerationRequest("id", "suite", 1, "case", "river", "step", 1, 2, 1.0, 1.0)
    with pytest.raises(ValueError, match="unknown cases"):
        run_generation_manifest(
            model, tokenizer, (request,), tmp_path / "out.jsonl", thoughts_by_case={"other": "x"}
        )
