from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from poetry50m.config import file_hash
from poetry50m.data.synthetic_sft import (
    FORM_LENGTH_LABELS,
    FORMAT_VERSION,
    GENERATION_OUTPUT_INSTRUCTION,
    LEGACY_RECIPE_VERSION,
    PROMPT_BLOCK_SIZE,
    PROMPT_CAPACITY,
    RECIPE_VERSION,
    SUBJECTS,
    TECHNIQUES,
    TONES,
    VOICES,
    PlannedExample,
    _observed_shape_prompt,
    _prompt_axis_indices,
    _validate_receipt_set,
    assemble_sft_dataset,
    finalize_sft_chunk,
    plan_sft_chunk,
    plan_uncapped_sft_retry,
    record_sft_dispatch,
    summarize_sft_chunks,
)
from poetry50m.data.tokenizer import save_tokenizer, train_tokenizer


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def write_results(path: Path, plan_path: Path, *, response_prefix: str = "Poem") -> None:
    plan = read_json(plan_path)
    examples = {
        example["example_id"]: example
        for example in plan["examples"]  # type: ignore[union-attr]
    }
    records = []
    for assignment in plan["assignments"]:  # type: ignore[union-attr]
        example_id = assignment["example_id"]
        line_count = examples[example_id]["prompt_spec"]["minimum_lines"]
        response = "\n".join(
            f"{response_prefix} {example_id} line {line_number}"
            for line_number in range(line_count)
        )
        records.append(
            {
                "custom_id": assignment["custom_id"],
                "response": {
                    "status_code": 200,
                    "body": {
                        "model": plan["model"],
                        "choices": [{"message": {"content": response}}],
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 20,
                            "total_tokens": 30,
                        },
                    },
                },
                "error": None,
            }
        )
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def record_dispatch(plan_path: Path) -> None:
    record_sft_dispatch(
        plan_path=plan_path,
        base_url="https://example.test/v1",
        api_key_environment_variable="TEST_API_KEY",
        concurrency=1,
        requests_per_minute=60,
        tokens_per_minute=100_000,
        timeout_seconds=30.0,
    )


@pytest.fixture
def tokenizer_path(tmp_path: Path) -> Path:
    tokenizer = train_tokenizer(
        (
            "Write a short original poem about rain and careful work.",
            "A precise first line\nA different and deliberate ending",
        )
    )
    path = tmp_path / "tokenizer.json"
    save_tokenizer(tokenizer, path)
    return path


def test_plans_are_deterministic_disjoint_and_model_scoped(tmp_path: Path) -> None:
    requests_a, plan_a = plan_sft_chunk(
        output_directory=tmp_path / "a",
        model="provider/model-a",
        provider="provider-a",
        start_index=0,
        example_count=9,
    )
    _, plan_b = plan_sft_chunk(
        output_directory=tmp_path / "b",
        model="provider/model-b",
        provider="provider-b",
        start_index=9,
        example_count=9,
    )
    value_a = read_json(plan_a)
    value_b = read_json(plan_b)

    assert value_a["chunk_id"] != value_b["chunk_id"]
    assert value_a["model"] == "provider/model-a"
    assert value_b["model"] == "provider/model-b"
    assert len(requests_a.read_text().splitlines()) == 9
    ids_a = {example["example_id"] for example in value_a["examples"]}  # type: ignore[union-attr]
    ids_b = {example["example_id"] for example in value_b["examples"]}  # type: ignore[union-attr]
    prompts_a = {example["prompt"] for example in value_a["examples"]}  # type: ignore[union-attr]
    assert ids_a.isdisjoint(ids_b)
    assert len(prompts_a) == 9
    assert PlannedExample.create(0, 20260728).prompt == value_a["examples"][0]["prompt"]  # type: ignore[index]


@pytest.mark.parametrize("start_index", (0, 4_096, PROMPT_CAPACITY - PROMPT_BLOCK_SIZE))
def test_production_blocks_are_balanced_and_form_length_compatible(start_index: int) -> None:
    specs = [
        PlannedExample.create(index, 20260728).prompt_spec
        for index in range(start_index, start_index + PROMPT_BLOCK_SIZE)
    ]

    for values, maximum_spread in (
        ((spec.subject for spec in specs), 1),
        (((spec.form, spec.length_label) for spec in specs), 1),
        ((spec.tone for spec in specs), 0),
        ((spec.voice for spec in specs), 0),
        ((spec.technique for spec in specs), 0),
    ):
        counts = Counter(values)
        assert max(counts.values()) - min(counts.values()) <= maximum_spread

    assert set(Counter(spec.subject for spec in specs)) == set(SUBJECTS)
    assert set(Counter(spec.tone for spec in specs)) == set(TONES)
    assert set(Counter(spec.voice for spec in specs)) == set(VOICES)
    assert set(Counter(spec.technique for spec in specs)) == set(TECHNIQUES)
    assert all(spec.length_label in FORM_LENGTH_LABELS[spec.form] for spec in specs)
    assert all(
        (spec.minimum_lines, spec.maximum_lines, spec.length_instruction)
        == (9, 9, "exactly 9 lines")
        for spec in specs
        if spec.form == "three restrained tercets"
    )


def test_prompt_interleaver_is_collision_free_across_full_capacity() -> None:
    combinations = {
        _prompt_axis_indices(index, 20260728) for index in range(PROMPT_CAPACITY)
    }
    assert len(combinations) == PROMPT_CAPACITY
    assert _prompt_axis_indices(0, 20260728) != _prompt_axis_indices(0, 20260729)


def test_observed_shape_prompt_and_mixed_recipe_receipts_are_explicit() -> None:
    prompt, prompt_spec = _observed_shape_prompt(
        {
            "subject": "rain",
            "tone": "quiet",
            "voice": "plain language",
        },
        line_count=11,
    )
    assert prompt.endswith("Use exactly 11 non-empty lines.")
    assert prompt_spec["minimum_lines"] == 11
    assert prompt_spec["maximum_lines"] == 11

    tokenizer_hash = "a" * 64
    base_receipt = {
        "format_version": FORMAT_VERSION,
        "tokenizer_sha256": tokenizer_hash,
        "seed": 20260728,
    }
    assert _validate_receipt_set(
        [
            {
                **base_receipt,
                "recipe_version": LEGACY_RECIPE_VERSION,
                "chunk_id": "legacy",
            },
            {
                **base_receipt,
                "recipe_version": RECIPE_VERSION,
                "chunk_id": "current",
            },
        ],
        expected_tokenizer_sha256=tokenizer_hash,
    ) == (tokenizer_hash, 20260728, (LEGACY_RECIPE_VERSION, RECIPE_VERSION))


def test_plan_refuses_overwrite_and_prompt_capacity_overflow(tmp_path: Path) -> None:
    arguments = {
        "output_directory": tmp_path / "chunk",
        "model": "model",
        "provider": "provider",
        "start_index": 0,
        "example_count": 1,
    }
    plan_sft_chunk(**arguments)
    with pytest.raises(FileExistsError):
        plan_sft_chunk(**arguments)
    with pytest.raises(ValueError, match="collision-free prompt capacity"):
        plan_sft_chunk(
            output_directory=tmp_path / "overflow",
            model="model",
            provider="provider",
            start_index=PROMPT_CAPACITY,
            example_count=1,
        )


def test_plan_requests_plain_poem_text_one_example_at_a_time(tmp_path: Path) -> None:
    requests_path, plan_path = plan_sft_chunk(
        output_directory=tmp_path / "chunk",
        model="model",
        provider="provider",
        start_index=0,
        example_count=1,
    )

    request = json.loads(requests_path.read_text().splitlines()[0])
    assert "response_format" not in request["body"]
    assert read_json(plan_path)["output_mode"] == "raw-text"
    assert request["body"]["messages"][0]["content"].endswith("Return only the poem text.")
    provider_prompt = request["body"]["messages"][1]["content"]
    training_prompt = read_json(plan_path)["examples"][0]["prompt"]
    assert provider_prompt == f"{training_prompt}\n\n{GENERATION_OUTPUT_INSTRUCTION}"
    assert GENERATION_OUTPUT_INSTRUCTION not in training_prompt


def test_plan_can_disable_provider_reasoning(tmp_path: Path) -> None:
    requests_path, plan_path = plan_sft_chunk(
        output_directory=tmp_path / "chunk",
        model="reasoning-model",
        provider="provider",
        start_index=0,
        example_count=1,
        reasoning_effort="none",
    )

    request = json.loads(requests_path.read_text().splitlines()[0])
    assert request["body"]["reasoning"] == {"effort": "none"}
    assert read_json(plan_path)["reasoning_effort"] == "none"


def test_retry_plan_selects_only_incomplete_responses_and_removes_limits(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _, plan_path = plan_sft_chunk(
        output_directory=source,
        model="reasoning-model",
        provider="provider",
        start_index=0,
        example_count=4,
        max_completion_tokens=1024,
        reasoning_effort="none",
    )
    write_results(source / "results.jsonl", plan_path)
    results = [
        json.loads(line) for line in (source / "results.jsonl").read_text().splitlines()
    ][:3]
    results[0]["response"]["body"]["choices"][0]["finish_reason"] = "stop"
    results[1]["response"]["body"]["choices"][0]["message"]["content"] = None
    results[1]["response"]["body"]["choices"][0]["finish_reason"] = "length"
    results[2]["response"]["body"]["choices"][0]["finish_reason"] = "length"
    (source / "results.jsonl").write_text(
        "".join(json.dumps(result) + "\n" for result in results),
        encoding="utf-8",
    )

    requests_path, retry_plan_path = plan_uncapped_sft_retry(
        source_plan_path=plan_path,
        source_results_path=source / "results.jsonl",
        output_directory=tmp_path / "retry",
    )

    requests = [json.loads(line) for line in requests_path.read_text().splitlines()]
    retry_plan = read_json(retry_plan_path)
    assert retry_plan["example_count"] == 3
    assert retry_plan["retry_reason_counts"] == {
        "completion_limit": 1,
        "missing_message_content": 1,
        "missing_result": 1,
    }
    assert all("max_completion_tokens" not in request["body"] for request in requests)
    assert all("max_tokens" not in request["body"] for request in requests)
    assert all("reasoning" not in request["body"] for request in requests)


def test_finalize_writes_sft_pairs_provenance_and_exact_counts(
    tmp_path: Path, tokenizer_path: Path
) -> None:
    chunk = tmp_path / "chunk"
    _, plan_path = plan_sft_chunk(
        output_directory=chunk,
        model="provider/model-a",
        provider="provider-a",
        start_index=20,
        example_count=3,
    )
    write_results(chunk / "results.jsonl", plan_path)
    record_dispatch(plan_path)

    receipt_path = finalize_sft_chunk(
        plan_path=plan_path,
        results_path=chunk / "results.jsonl",
        tokenizer_path=tokenizer_path,
        output_directory=tmp_path / "finalized",
        expected_tokenizer_sha256=file_hash(tokenizer_path),
    )

    receipt = read_json(receipt_path)
    records = [
        json.loads(line)
        for line in (tmp_path / "finalized" / "examples.jsonl").read_text().splitlines()
    ]
    assert receipt["example_count"] == 3
    assert receipt["model"] == "provider/model-a"
    assert receipt["observed_models"] == ["provider/model-a"]
    assert receipt["missing_observed_model_records"] == 0
    assert receipt["provider_usage"] == {
        "prompt_tokens": 30,
        "completion_tokens": 60,
        "total_tokens": 90,
        "missing_usage_records": 0,
    }
    assert receipt["token_counts"]["formatted"] > receipt["token_counts"]["supervised"]  # type: ignore[index]
    assert records[0]["messages"][0]["role"] == "user"
    assert records[0]["messages"][1]["role"] == "assistant"
    assert records[0]["provenance"]["kind"] == "synthetic"
    assert records[0]["provenance"]["generator_model"] == "provider/model-a"
    assert records[0]["token_counts"]["supervised"] == records[0]["token_counts"]["response"] + 1
    assert records[0]["token_counts"]["training_input"] == (
        records[0]["token_counts"]["formatted"] - 1
    )
    assert records[0]["token_counts"]["formatted"] == (
        records[0]["token_counts"]["prompt"] + records[0]["token_counts"]["response"] + 4
    )


def test_finalize_fails_closed_on_duplicate_results(tmp_path: Path, tokenizer_path: Path) -> None:
    chunk = tmp_path / "chunk"
    _, plan_path = plan_sft_chunk(
        output_directory=chunk,
        model="model",
        provider="provider",
        start_index=0,
        example_count=2,
    )
    write_results(chunk / "results.jsonl", plan_path)
    record_dispatch(plan_path)
    first_line = (chunk / "results.jsonl").read_text().splitlines()[0]
    (chunk / "results.jsonl").write_text(f"{first_line}\n{first_line}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate result"):
        finalize_sft_chunk(
            plan_path=plan_path,
            results_path=chunk / "results.jsonl",
            tokenizer_path=tokenizer_path,
            output_directory=tmp_path / "finalized",
            expected_tokenizer_sha256=file_hash(tokenizer_path),
        )


def test_finalize_can_preserve_an_explicitly_partial_paid_chunk(
    tmp_path: Path, tokenizer_path: Path
) -> None:
    chunk = tmp_path / "chunk"
    _, plan_path = plan_sft_chunk(
        output_directory=chunk,
        model="model",
        provider="provider",
        start_index=0,
        example_count=3,
    )
    write_results(chunk / "results.jsonl", plan_path)
    record_dispatch(plan_path)
    completed = (chunk / "results.jsonl").read_text().splitlines()[:2]
    (chunk / "results.jsonl").write_text("\n".join(completed) + "\n", encoding="utf-8")

    receipt_path = finalize_sft_chunk(
        plan_path=plan_path,
        results_path=chunk / "results.jsonl",
        tokenizer_path=tokenizer_path,
        output_directory=tmp_path / "finalized",
        expected_tokenizer_sha256=file_hash(tokenizer_path),
        allow_partial=True,
    )

    receipt = read_json(receipt_path)
    assert receipt["planned_example_count"] == 3
    assert receipt["completed_result_count"] == 2
    assert receipt["missing_result_count"] == 1
    assert receipt["partial"] is True
    assert receipt["example_count"] == 2


def test_finalize_records_unusable_paid_responses_without_aborting(
    tmp_path: Path, tokenizer_path: Path
) -> None:
    chunk = tmp_path / "chunk"
    _, plan_path = plan_sft_chunk(
        output_directory=chunk,
        model="model",
        provider="provider",
        start_index=0,
        example_count=2,
    )
    write_results(chunk / "results.jsonl", plan_path)
    record_dispatch(plan_path)
    results = [json.loads(line) for line in (chunk / "results.jsonl").read_text().splitlines()]
    message = results[1]["response"]["body"]["choices"][0]["message"]
    message["content"] = None
    message["reasoning"] = "provider reasoning without a final message"
    (chunk / "results.jsonl").write_text(
        "".join(json.dumps(result) + "\n" for result in results),
        encoding="utf-8",
    )

    receipt_path = finalize_sft_chunk(
        plan_path=plan_path,
        results_path=chunk / "results.jsonl",
        tokenizer_path=tokenizer_path,
        output_directory=tmp_path / "finalized",
        expected_tokenizer_sha256=file_hash(tokenizer_path),
    )

    receipt = read_json(receipt_path)
    rejection = json.loads(
        (tmp_path / "finalized" / "rejections.jsonl").read_text().splitlines()[0]
    )
    assert receipt["completed_result_count"] == 2
    assert receipt["usable_result_count"] == 1
    assert receipt["unusable_result_count"] == 1
    assert receipt["example_count"] == 1
    assert rejection["reasons"] == ["unusable_provider_response"]


def test_finalize_rejects_length_stopped_responses_for_uncapped_retry(
    tmp_path: Path, tokenizer_path: Path
) -> None:
    chunk = tmp_path / "chunk"
    _, plan_path = plan_sft_chunk(
        output_directory=chunk,
        model="model",
        provider="provider",
        start_index=0,
        example_count=1,
    )
    write_results(chunk / "results.jsonl", plan_path)
    record_dispatch(plan_path)
    result = json.loads((chunk / "results.jsonl").read_text())
    result["response"]["body"]["choices"][0]["finish_reason"] = "length"
    (chunk / "results.jsonl").write_text(json.dumps(result) + "\n", encoding="utf-8")

    receipt_path = finalize_sft_chunk(
        plan_path=plan_path,
        results_path=chunk / "results.jsonl",
        tokenizer_path=tokenizer_path,
        output_directory=tmp_path / "finalized",
        expected_tokenizer_sha256=file_hash(tokenizer_path),
    )

    receipt = read_json(receipt_path)
    rejection = json.loads(
        (tmp_path / "finalized" / "rejections.jsonl").read_text().splitlines()[0]
    )
    assert receipt["truncated_result_count"] == 1
    assert receipt["example_count"] == 0
    assert rejection["reasons"] == ["completion_limit"]


def test_finalize_keeps_off_length_poems_with_an_adjusted_prompt(
    tmp_path: Path, tokenizer_path: Path
) -> None:
    chunk = tmp_path / "chunk"
    _, plan_path = plan_sft_chunk(
        output_directory=chunk,
        model="model",
        provider="provider",
        start_index=0,
        example_count=1,
    )
    write_results(chunk / "results.jsonl", plan_path)
    record_dispatch(plan_path)
    result = json.loads((chunk / "results.jsonl").read_text())
    plan = read_json(plan_path)
    examples = plan.get("examples")
    assert isinstance(examples, list)
    prompt_spec = examples[0]["prompt_spec"]
    assert isinstance(prompt_spec, dict)
    maximum_lines = prompt_spec.get("maximum_lines")
    assert isinstance(maximum_lines, int)
    message = result["response"]["body"]["choices"][0]["message"]
    message["content"] = "\n".join(
        f"A useful generated line {index}" for index in range(maximum_lines + 1)
    )
    (chunk / "results.jsonl").write_text(json.dumps(result) + "\n", encoding="utf-8")

    receipt_path = finalize_sft_chunk(
        plan_path=plan_path,
        results_path=chunk / "results.jsonl",
        tokenizer_path=tokenizer_path,
        output_directory=tmp_path / "finalized",
        expected_tokenizer_sha256=file_hash(tokenizer_path),
    )

    receipt = read_json(receipt_path)
    example = json.loads(
        (tmp_path / "finalized" / "examples.jsonl").read_text().splitlines()[0]
    )
    assert receipt["example_count"] == 1
    assert receipt["rejected_example_count"] == 0
    assert receipt["adjusted_prompt_count"] == 1
    assert example["provenance"]["prompt_adjustment"]["kind"] == "observed-shape"
    assert "Use exactly" in example["messages"][0]["content"]


def test_finalize_strips_poem_markdown_and_records_transformations(
    tmp_path: Path, tokenizer_path: Path
) -> None:
    chunk = tmp_path / "chunk"
    _, plan_path = plan_sft_chunk(
        output_directory=chunk,
        model="model",
        provider="provider",
        start_index=0,
        example_count=1,
    )
    write_results(chunk / "results.jsonl", plan_path)
    record_dispatch(plan_path)
    result = json.loads((chunk / "results.jsonl").read_text())
    result["response"]["body"]["choices"][0]["message"]["content"] = (
        "*First line*\nSecond line\n***\nThird *line\nFourth line"
    )
    (chunk / "results.jsonl").write_text(json.dumps(result) + "\n", encoding="utf-8")

    receipt_path = finalize_sft_chunk(
        plan_path=plan_path,
        results_path=chunk / "results.jsonl",
        tokenizer_path=tokenizer_path,
        output_directory=tmp_path / "finalized",
        expected_tokenizer_sha256=file_hash(tokenizer_path),
    )

    receipt = read_json(receipt_path)
    example = json.loads(
        (tmp_path / "finalized" / "examples.jsonl").read_text().splitlines()[0]
    )
    assert example["messages"][1]["content"] == (
        "First line\nSecond line\n\nThird line\nFourth line"
    )
    assert receipt["markdown_sanitized_count"] == 1
    assert receipt["markdown_transformations"] == {
        "asterisk": 1,
        "horizontal_rule": 1,
        "italic_delimiters": 1,
    }


def test_finalize_relabels_incorrect_stanza_structure(
    tmp_path: Path, tokenizer_path: Path
) -> None:
    chunk = tmp_path / "chunk"
    _, plan_path = plan_sft_chunk(
        output_directory=chunk,
        model="model",
        provider="provider",
        start_index=6166,
        example_count=1,
    )
    write_results(chunk / "results.jsonl", plan_path)
    record_dispatch(plan_path)
    plan = read_json(plan_path)
    assert plan["examples"][0]["prompt_spec"]["form"] == (
        "two unequal stanzas separated by a one-line turn"
    )
    line_count = plan["examples"][0]["prompt_spec"]["minimum_lines"]
    result = json.loads((chunk / "results.jsonl").read_text())
    result["response"]["body"]["choices"][0]["message"]["content"] = "\n".join(
        f"Single stanza line {index}" for index in range(line_count)
    )
    (chunk / "results.jsonl").write_text(json.dumps(result) + "\n", encoding="utf-8")

    finalize_sft_chunk(
        plan_path=plan_path,
        results_path=chunk / "results.jsonl",
        tokenizer_path=tokenizer_path,
        output_directory=tmp_path / "finalized",
        expected_tokenizer_sha256=file_hash(tokenizer_path),
    )

    example = json.loads(
        (tmp_path / "finalized" / "examples.jsonl").read_text().splitlines()[0]
    )
    assert example["prompt_spec"]["form"] == "an original English poem"
    assert example["provenance"]["prompt_adjustment"]["reasons"] == ["stanza_structure"]


def test_finalize_normalizes_newlines_and_records_local_rejections(
    tmp_path: Path, tokenizer_path: Path
) -> None:
    chunk = tmp_path / "chunk"
    _, plan_path = plan_sft_chunk(
        output_directory=chunk,
        model="model",
        provider="provider",
        start_index=0,
        example_count=2,
    )
    write_results(chunk / "results.jsonl", plan_path)
    record_dispatch(plan_path)
    results = [json.loads(line) for line in (chunk / "results.jsonl").read_text().splitlines()]
    first_message = results[0]["response"]["body"]["choices"][0]["message"]
    first_message["content"] = first_message["content"].replace("\n", "\r\n")
    results[1]["response"]["body"]["choices"][0]["message"]["content"] = (
        "As an AI, I cannot fulfill this request."
    )
    (chunk / "results.jsonl").write_text(
        "".join(json.dumps(result) + "\n" for result in results), encoding="utf-8"
    )

    receipt_path = finalize_sft_chunk(
        plan_path=plan_path,
        results_path=chunk / "results.jsonl",
        tokenizer_path=tokenizer_path,
        output_directory=tmp_path / "finalized",
        expected_tokenizer_sha256=file_hash(tokenizer_path),
    )

    receipt = read_json(receipt_path)
    examples_text = (tmp_path / "finalized" / "examples.jsonl").read_text()
    rejections = [
        json.loads(line)
        for line in (tmp_path / "finalized" / "rejections.jsonl").read_text().splitlines()
    ]
    assert receipt["example_count"] == 1
    assert receipt["rejected_example_count"] == 1
    assert "\r" not in examples_text
    assert "refusal_boilerplate" in rejections[0]["reasons"]


def finalized_chunk(
    tmp_path: Path,
    tokenizer_path: Path,
    *,
    name: str,
    model: str,
    start_index: int,
    response_prefix: str,
) -> Path:
    chunk = tmp_path / f"{name}-raw"
    _, plan_path = plan_sft_chunk(
        output_directory=chunk,
        model=model,
        provider="test-provider",
        start_index=start_index,
        example_count=2,
    )
    write_results(chunk / "results.jsonl", plan_path, response_prefix=response_prefix)
    record_dispatch(plan_path)
    output = tmp_path / name
    return finalize_sft_chunk(
        plan_path=plan_path,
        results_path=chunk / "results.jsonl",
        tokenizer_path=tokenizer_path,
        output_directory=output,
        expected_tokenizer_sha256=file_hash(tokenizer_path),
    )


def test_summary_and_assembly_combine_multiple_models_deterministically(
    tmp_path: Path, tokenizer_path: Path
) -> None:
    receipt_b = finalized_chunk(
        tmp_path,
        tokenizer_path,
        name="b",
        model="model-b",
        start_index=2,
        response_prefix="Second",
    )
    receipt_a = finalized_chunk(
        tmp_path,
        tokenizer_path,
        name="a",
        model="model-a",
        start_index=0,
        response_prefix="First",
    )

    summary = summarize_sft_chunks(
        [receipt_b, receipt_a],
        target_tokens=1_000_000,
        expected_tokenizer_sha256=file_hash(tokenizer_path),
    )
    assert summary["example_count"] == 4
    assert summary["examples_by_model"] == {"model-a": 2, "model-b": 2}
    assert summary["remaining_formatted_tokens"] > 0

    assembly_receipt = assemble_sft_dataset(
        [receipt_b, receipt_a],
        output_directory=tmp_path / "assembled",
        target_tokens=1,
        target_metric="supervised",
        expected_tokenizer_sha256=file_hash(tokenizer_path),
    )
    assembly = read_json(assembly_receipt)
    dataset = [
        json.loads(line)
        for line in (tmp_path / "assembled" / "dataset.jsonl").read_text().splitlines()
    ]
    assert assembly["target_reached"] is True
    assert assembly["example_count"] == 1
    assert dataset[0]["example_id"] == "synthetic-sft-v2-000000000"


def test_assembly_fails_closed_when_validated_data_is_under_target(
    tmp_path: Path, tokenizer_path: Path
) -> None:
    receipt = finalized_chunk(
        tmp_path,
        tokenizer_path,
        name="partial",
        model="model-a",
        start_index=0,
        response_prefix="First",
    )
    with pytest.raises(ValueError, match="below the required"):
        assemble_sft_dataset(
            [receipt],
            output_directory=tmp_path / "assembled",
            target_tokens=1_000_000,
            expected_tokenizer_sha256=file_hash(tokenizer_path),
        )
    assert not (tmp_path / "assembled").exists()


def test_assembly_rejects_overlap_after_an_early_token_cutoff(
    tmp_path: Path, tokenizer_path: Path
) -> None:
    receipt_a = finalized_chunk(
        tmp_path,
        tokenizer_path,
        name="a",
        model="model-a",
        start_index=0,
        response_prefix="First",
    )
    receipt_b = finalized_chunk(
        tmp_path,
        tokenizer_path,
        name="b",
        model="model-b",
        start_index=1,
        response_prefix="Second",
    )
    with pytest.raises(ValueError, match="overlapping chunks"):
        assemble_sft_dataset(
            [receipt_a, receipt_b],
            output_directory=tmp_path / "assembled",
            target_tokens=1,
            expected_tokenizer_sha256=file_hash(tokenizer_path),
        )
