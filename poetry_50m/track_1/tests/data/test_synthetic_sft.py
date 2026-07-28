from __future__ import annotations

import json
from pathlib import Path

import pytest

from poetry50m.config import file_hash
from poetry50m.data.synthetic_sft import (
    PROMPT_CAPACITY,
    PlannedExample,
    assemble_sft_dataset,
    finalize_sft_chunk,
    plan_sft_chunk,
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
        responses = [
            {
                "example_id": example_id,
                "response": "\n".join(
                    [
                        f"{response_prefix} {example_id} line {line_number}"
                        for line_number in range(
                            {
                                "very short": 4,
                                "short": 8,
                                "medium": 13,
                                "long": 21,
                            }[examples[example_id]["prompt_spec"]["length_label"]]
                        )
                    ]
                ),
            }
            for example_id in assignment["example_ids"]
            if example_id in examples
        ]
        records.append(
            {
                "custom_id": assignment["custom_id"],
                "response": {
                    "status_code": 200,
                    "body": {
                        "model": plan["model"],
                        "choices": [{"message": {"content": json.dumps({"responses": responses})}}],
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
        examples_per_request=4,
    )
    _, plan_b = plan_sft_chunk(
        output_directory=tmp_path / "b",
        model="provider/model-b",
        provider="provider-b",
        start_index=9,
        example_count=9,
        examples_per_request=4,
    )
    value_a = read_json(plan_a)
    value_b = read_json(plan_b)

    assert value_a["chunk_id"] != value_b["chunk_id"]
    assert value_a["model"] == "provider/model-a"
    assert value_b["model"] == "provider/model-b"
    assert len(requests_a.read_text().splitlines()) == 3
    ids_a = {example["example_id"] for example in value_a["examples"]}  # type: ignore[union-attr]
    ids_b = {example["example_id"] for example in value_b["examples"]}  # type: ignore[union-attr]
    prompts_a = {example["prompt"] for example in value_a["examples"]}  # type: ignore[union-attr]
    assert ids_a.isdisjoint(ids_b)
    assert len(prompts_a) == 9
    assert PlannedExample.create(0, 20260728).prompt == value_a["examples"][0]["prompt"]  # type: ignore[index]


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


def test_plan_can_omit_unsupported_response_format_parameter(tmp_path: Path) -> None:
    requests_path, plan_path = plan_sft_chunk(
        output_directory=tmp_path / "chunk",
        model="model",
        provider="provider",
        start_index=0,
        example_count=1,
        response_format_mode="none",
    )

    request = json.loads(requests_path.read_text().splitlines()[0])
    assert "response_format" not in request["body"]
    assert read_json(plan_path)["response_format_mode"] == "none"
    assert '{"responses":' in request["body"]["messages"][0]["content"]


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
        examples_per_request=2,
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
        "prompt_tokens": 20,
        "completion_tokens": 40,
        "total_tokens": 60,
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


def test_finalize_fails_closed_on_missing_or_reordered_outputs(
    tmp_path: Path, tokenizer_path: Path
) -> None:
    chunk = tmp_path / "chunk"
    _, plan_path = plan_sft_chunk(
        output_directory=chunk,
        model="model",
        provider="provider",
        start_index=0,
        example_count=2,
        examples_per_request=2,
    )
    write_results(chunk / "results.jsonl", plan_path)
    record_dispatch(plan_path)
    result = json.loads((chunk / "results.jsonl").read_text().splitlines()[0])
    content = json.loads(result["response"]["body"]["choices"][0]["message"]["content"])
    content["responses"].reverse()
    result["response"]["body"]["choices"][0]["message"]["content"] = json.dumps(content)
    (chunk / "results.jsonl").write_text(json.dumps(result) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="IDs or order"):
        finalize_sft_chunk(
            plan_path=plan_path,
            results_path=chunk / "results.jsonl",
            tokenizer_path=tokenizer_path,
            output_directory=tmp_path / "finalized",
            expected_tokenizer_sha256=file_hash(tokenizer_path),
        )


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
        examples_per_request=2,
    )
    write_results(chunk / "results.jsonl", plan_path)
    record_dispatch(plan_path)
    result = json.loads((chunk / "results.jsonl").read_text().splitlines()[0])
    content = json.loads(result["response"]["body"]["choices"][0]["message"]["content"])
    content["responses"][0]["response"] = content["responses"][0]["response"].replace("\n", "\r\n")
    content["responses"][1]["response"] = "I cannot write that poem."
    result["response"]["body"]["choices"][0]["message"]["content"] = json.dumps(content)
    (chunk / "results.jsonl").write_text(json.dumps(result) + "\n", encoding="utf-8")

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
        examples_per_request=2,
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
    assert dataset[0]["example_id"] == "synthetic-sft-000000000"


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
