from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pytest

from poetry50m.data.artifacts import (
    read_prompt_records,
    write_pairings,
    write_prompt_records,
    write_thought_records,
)
from poetry50m.data.loaders import iter_manifest, write_manifest
from poetry50m.data.schema import ContentBlock, Provenance, SourceDocument
from poetry50m.data.synthetic_corpus import (
    SyntheticCorpusConfig,
    _final_text_only_response_body,
    finalize_synthetic_corpus,
    ingest_generation_results,
    merge_corpus_artifacts,
    plan_generation,
    run_openai_compatible_batch,
)


def test_final_text_storage_drops_reasoning_and_rejects_reasoning_only() -> None:
    stored = _final_text_only_response_body(
        {
            "model": "provider/model",
            "choices": [
                {
                    "message": {
                        "content": "A clean poem",
                        "reasoning": "Private working",
                    }
                }
            ],
            "usage": {"total_tokens": 12},
        },
        custom_id="request-1",
    )

    assert stored == {
        "model": "provider/model",
        "choices": [{"message": {"role": "assistant", "content": "A clean poem"}}],
        "usage": {"total_tokens": 12},
    }
    with pytest.raises(ValueError, match="returned reasoning without final content"):
        _final_text_only_response_body(
            {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "reasoning": "A draft without a final answer",
                        }
                    }
                ]
            },
            custom_id="request-2",
        )


def config_path() -> Path:
    return Path(__file__).parents[2] / "configs" / "data" / "synthetic_cerebras_8m_v1.json"


def write_jsonl(path: Path, values: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


def completion_result(custom_id: str, content: dict[str, object]) -> dict[str, object]:
    return {
        "custom_id": custom_id,
        "response": {
            "status_code": 200,
            "body": {"choices": [{"message": {"content": json.dumps(content)}}]},
        },
        "error": None,
    }


def candidate(index: int, *, repeated: bool = False) -> dict[str, object]:
    lines = (
        ["The kettle clicks beside the rain"] * 8
        if repeated
        else [
            f"A bicycle{index} bell{index} crosses{index} wet{index} stone{index}",
            f"Bread{index} cools{index} behind{index} the bakery{index} glass{index}",
            f"A courier{index} folds{index} the morning{index} map{index}",
            f"Pigeons{index} lift{index} around{index} a yellow{index} bus{index}",
            f"Someone{index} laughs{index} beneath{index} an awning{index}",
            f"The gutter{index} carries{index} one red{index} leaf{index}",
            f"Traffic{index} loosens{index} into separate{index} voices{index}",
            f"And every{index} doorway{index} keeps{index} its weather{index}",
        ]
    )
    poem = "\n".join(lines)
    filler = " ".join(f"candidate{index}word{number}" for number in range(70))
    return {
        "title": f"City Weather {index}",
        "prompts": [
            {"text": f"Write about a changing city morning {index}.", "method": "theme"},
            {"text": "Use a bicycle bell, bread, and rain.", "method": "imagery"},
            {"text": "Make an ordinary commute feel newly visible.", "method": "paraphrase"},
        ],
        "poem": f"{poem}\n{filler}",
        "themes": ["attention", "change"],
        "imagery": ["bicycle bell", "wet stone", "bread"],
        "mood": "alert",
        "form": "free verse",
    }


def accepted_critique() -> dict[str, object]:
    return {
        "prompt_adherence": 5,
        "coherence": 5,
        "craft": 4,
        "originality": 4,
        "degeneration": False,
        "named_author_imitation": False,
        "suspected_quote": False,
        "decision": "accept",
        "reasons": ["Specific, coherent, and prompt-conditioned."],
    }


def test_plan_assigns_diverse_lanes_and_strict_schemas(tmp_path: Path) -> None:
    requests_path, plan_path = plan_generation(
        config_path(), request_count=4, output_directory=tmp_path
    )
    requests = [json.loads(line) for line in requests_path.read_text().splitlines()]
    assert [request["body"]["model"] for request in requests] == [
        "gpt-oss-120b",
        "gpt-oss-120b",
        "gpt-oss-120b",
        "gpt-oss-120b",
    ]
    assert [request["body"]["temperature"] for request in requests] == [
        0.85,
        0.85,
        1.15,
        0.95,
    ]
    assert [request["body"]["seed"] for request in requests] == [
        20260727,
        20260728,
        20260729,
        20260730,
    ]
    assert all(
        request["body"]["response_format"]["json_schema"]["strict"] is True for request in requests
    )
    assert "minItems" not in json.dumps(requests)
    assert "maxItems" not in json.dumps(requests)
    plan = json.loads(plan_path.read_text())
    assert plan["candidate_capacity"] == 16
    assert all(len(assignment["briefs"]) == 4 for assignment in plan["assignments"])
    assert all(
        set(brief)
        == {
            "setting",
            "required_objects",
            "physical_event",
            "emotional_pressure",
            "participants",
            "form",
        }
        for assignment in plan["assignments"]
        for brief in assignment["briefs"]
    )


def test_plan_supports_openai_compatible_request_shapes(tmp_path: Path) -> None:
    requests_path, plan_path = plan_generation(
        config_path(),
        request_count=1,
        output_directory=tmp_path,
        model_override="provider/good-model",
        openai_compatible=True,
        response_format_mode="json-object",
        max_tokens_field="max_tokens",
    )

    request = json.loads(requests_path.read_text().splitlines()[0])
    body = request["body"]
    assert body["model"] == "provider/good-model"
    assert body["max_tokens"] == 4096
    assert "max_completion_tokens" not in body
    assert "reasoning_effort" not in body
    assert "seed" not in body
    assert body["response_format"] == {"type": "json_object"}

    plan = json.loads(plan_path.read_text())
    assert plan["openai_compatible"] is True
    assert plan["response_format_mode"] == "json-object"
    assert plan["max_tokens_field"] == "max_tokens"


def test_openai_compatible_runner_posts_and_preserves_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests_path, _ = plan_generation(
        config_path(),
        request_count=1,
        output_directory=tmp_path / "plan",
        model_override="provider/good-model",
        openai_compatible=True,
    )
    results_path = tmp_path / "generation.results.jsonl"
    captured: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        @staticmethod
        def getcode() -> int:
            return 200

        @staticmethod
        def read() -> bytes:
            return json.dumps(
                {
                    "choices": [{"message": {"content": '{"examples":[]}'}}],
                    "usage": {"total_tokens": 10},
                }
            ).encode()

    def fake_urlopen(request: urllib.request.Request, *, timeout: float) -> FakeResponse:
        captured["url"] = request.full_url
        captured["authorization"] = request.get_header("Authorization")
        captured["body"] = json.loads(request.data)
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setenv("SYNTH_API_KEY", "test-secret")
    monkeypatch.setattr(
        "poetry50m.data.synthetic_corpus.urllib.request.urlopen",
        fake_urlopen,
    )

    run_openai_compatible_batch(
        requests_path,
        results_path,
        base_url="https://example.test/v1/",
        api_key_environment_variable="SYNTH_API_KEY",
        concurrency=1,
        requests_per_minute=1,
        tokens_per_minute=10_000,
        timeout_seconds=42.0,
    )

    result = json.loads(results_path.read_text().splitlines()[0])
    assert result["custom_id"] == "poetry-synthetic-00000000"
    assert result["response"]["status_code"] == 200
    assert captured["url"] == "https://example.test/v1/chat/completions"
    assert captured["authorization"] == "Bearer test-secret"
    assert captured["body"]["model"] == "provider/good-model"
    assert captured["timeout"] == 42.0


def test_ingest_and_finalize_write_synthetic_corpus_contracts(tmp_path: Path) -> None:
    requests_path, _ = plan_generation(
        config_path(), request_count=1, output_directory=tmp_path / "plan"
    )
    generation_results = tmp_path / "generation.results.jsonl"
    write_jsonl(
        generation_results,
        [
            completion_result(
                "poetry-synthetic-00000000",
                {"examples": [candidate(index) for index in range(4)]},
            )
        ],
    )
    candidates_path, critic_requests_path = ingest_generation_results(
        config_path(),
        requests_path=requests_path,
        results_path=generation_results,
        output_directory=tmp_path / "ingested",
    )
    stored_candidates = [json.loads(line) for line in candidates_path.read_text().splitlines()]
    critic_requests = [json.loads(line) for line in critic_requests_path.read_text().splitlines()]
    critic_results = tmp_path / "critic.results.jsonl"
    write_jsonl(
        critic_results,
        [
            completion_result(
                request["custom_id"],
                accepted_critique(),
            )
            for request in critic_requests
        ],
    )

    receipt_path = finalize_synthetic_corpus(
        config_path(),
        candidates_path=candidates_path,
        critic_results_path=critic_results,
        output_directory=tmp_path / "corpus",
    )

    receipt = json.loads(receipt_path.read_text())
    documents = tuple(iter_manifest(tmp_path / "corpus" / "manifest.jsonl", allow_synthetic=True))
    prompts = read_prompt_records(tmp_path / "corpus" / "prompts.jsonl")
    assert len(stored_candidates) == 4
    assert receipt["accepted_count"] == 4
    assert len(documents) == 4
    assert all(document.provenance.rights_status == "synthetic" for document in documents)
    assert len(prompts) == 12
    with pytest.raises(PermissionError, match="allow_synthetic"):
        tuple(iter_manifest(tmp_path / "corpus" / "manifest.jsonl"))

    base = tmp_path / "base"
    base.mkdir()
    write_manifest(base / "manifest.jsonl", ())
    write_prompt_records(base / "prompts.jsonl", ())
    write_thought_records(base / "thoughts.jsonl", ())
    write_pairings(base / "pairings.jsonl", ())
    merge_receipt_path = merge_corpus_artifacts(
        base_manifest=base / "manifest.jsonl",
        base_prompts=base / "prompts.jsonl",
        base_thoughts=base / "thoughts.jsonl",
        base_pairings=base / "pairings.jsonl",
        synthetic_directory=tmp_path / "corpus",
        output_directory=tmp_path / "merged",
    )
    merge_receipt = json.loads(merge_receipt_path.read_text())
    assert merge_receipt["counts"] == {
        "documents": 4,
        "pairings": 0,
        "prompts": 12,
        "thoughts": 0,
    }


def test_ingest_and_finalize_without_critic_uses_local_gates(tmp_path: Path) -> None:
    requests_path, _ = plan_generation(
        config_path(),
        request_count=1,
        output_directory=tmp_path / "plan",
        model_override="provider/good-model",
        openai_compatible=True,
    )
    generation_results = tmp_path / "generation.results.jsonl"
    write_jsonl(
        generation_results,
        [
            completion_result(
                "poetry-synthetic-00000000",
                {"examples": [candidate(index) for index in range(4)]},
            )
        ],
    )
    candidates_path, critic_requests_path = ingest_generation_results(
        config_path(),
        requests_path=requests_path,
        results_path=generation_results,
        output_directory=tmp_path / "ingested",
        create_critic_requests=False,
    )

    assert critic_requests_path.read_text() == ""
    ingest_receipt = json.loads(
        (tmp_path / "ingested" / "generation.ingest.receipt.json").read_text()
    )
    assert ingest_receipt["critic_requests_enabled"] is False

    receipt_path = finalize_synthetic_corpus(
        config_path(),
        candidates_path=candidates_path,
        critic_results_path=None,
        output_directory=tmp_path / "corpus",
    )

    receipt = json.loads(receipt_path.read_text())
    quality = [
        json.loads(line)
        for line in (tmp_path / "corpus" / "quality.jsonl").read_text().splitlines()
    ]
    documents = tuple(iter_manifest(tmp_path / "corpus" / "manifest.jsonl", allow_synthetic=True))
    assert receipt["accepted_count"] == 4
    assert receipt["critic_enabled"] is False
    assert receipt["critic_results_sha256"] is None
    assert receipt["critic_usage"] is None
    assert all(row["critic"] is None for row in quality)
    assert all("critic_model" not in document.metadata for document in documents)
    assert all(
        document.transformation_lineage
        == ("openai_compatible_json_generation", "local_quality_gates")
        for document in documents
    )


def test_ingest_preserves_valid_candidates_and_receipts_malformed_results(
    tmp_path: Path,
) -> None:
    requests_path, _ = plan_generation(
        config_path(), request_count=2, output_directory=tmp_path / "plan"
    )
    examples = [candidate(index) for index in range(4)]
    examples[0]["themes"] = ["attention"]
    generation_results = tmp_path / "generation.results.jsonl"
    write_jsonl(
        generation_results,
        [
            completion_result(
                "poetry-synthetic-00000000",
                {"examples": examples},
            ),
            {
                "custom_id": "poetry-synthetic-00000001",
                "response": {
                    "status_code": 200,
                    "body": {"choices": [{"message": {"content": '{"examples":['}}]},
                },
                "error": None,
            },
        ],
    )

    candidates_path, critic_requests_path = ingest_generation_results(
        config_path(),
        requests_path=requests_path,
        results_path=generation_results,
        output_directory=tmp_path / "ingested",
    )

    candidates = [json.loads(line) for line in candidates_path.read_text().splitlines()]
    critic_requests = [json.loads(line) for line in critic_requests_path.read_text().splitlines()]
    receipt = json.loads((tmp_path / "ingested" / "generation.ingest.receipt.json").read_text())
    assert len(candidates) == 4
    assert len(critic_requests) == 4
    assert candidates[0]["themes"] == ["attention"]
    assert receipt["candidate_rejections"] == []
    assert receipt["generation_result_rejections"][0]["custom_id"] == ("poetry-synthetic-00000001")


def test_finalize_rejects_local_repetition_even_when_critic_accepts(tmp_path: Path) -> None:
    requests_path, _ = plan_generation(
        config_path(), request_count=1, output_directory=tmp_path / "plan"
    )
    generation_results = tmp_path / "generation.results.jsonl"
    write_jsonl(
        generation_results,
        [
            completion_result(
                "poetry-synthetic-00000000",
                {"examples": [candidate(0, repeated=True), *[candidate(i) for i in range(1, 4)]]},
            )
        ],
    )
    candidates_path, critic_requests_path = ingest_generation_results(
        config_path(),
        requests_path=requests_path,
        results_path=generation_results,
        output_directory=tmp_path / "ingested",
    )
    critic_requests = [json.loads(line) for line in critic_requests_path.read_text().splitlines()]
    critic_results = tmp_path / "critic.results.jsonl"
    write_jsonl(
        critic_results,
        [
            completion_result(request["custom_id"], accepted_critique())
            for request in critic_requests
        ],
    )

    receipt_path = finalize_synthetic_corpus(
        config_path(),
        candidates_path=candidates_path,
        critic_results_path=critic_results,
        output_directory=tmp_path / "corpus",
    )

    receipt = json.loads(receipt_path.read_text())
    quality_path = tmp_path / "corpus" / "quality.jsonl"
    quality = [json.loads(line) for line in quality_path.read_text().splitlines()]
    assert receipt["accepted_count"] == 3
    assert any("repeated_lines" in row["rejection_reasons"] for row in quality)


def test_finalize_rejects_stock_language_and_markdown_artifacts(tmp_path: Path) -> None:
    requests_path, _ = plan_generation(
        config_path(), request_count=1, output_directory=tmp_path / "plan"
    )
    examples = [candidate(index) for index in range(4)]
    examples[0]["poem"] = str(examples[0]["poem"]).replace(
        "A bicycle0 bell0 crosses0 wet0 stone0",
        'A silver thread whispers across wet stone"\\',
    )
    generation_results = tmp_path / "generation.results.jsonl"
    write_jsonl(
        generation_results,
        [
            completion_result(
                "poetry-synthetic-00000000",
                {"examples": examples},
            )
        ],
    )
    candidates_path, critic_requests_path = ingest_generation_results(
        config_path(),
        requests_path=requests_path,
        results_path=generation_results,
        output_directory=tmp_path / "ingested",
    )
    critic_requests = [json.loads(line) for line in critic_requests_path.read_text().splitlines()]
    critic_results = tmp_path / "critic.results.jsonl"
    write_jsonl(
        critic_results,
        [
            completion_result(request["custom_id"], accepted_critique())
            for request in critic_requests
        ],
    )

    receipt_path = finalize_synthetic_corpus(
        config_path(),
        candidates_path=candidates_path,
        critic_results_path=critic_results,
        output_directory=tmp_path / "corpus",
    )

    receipt = json.loads(receipt_path.read_text())
    quality = [
        json.loads(line)
        for line in (tmp_path / "corpus" / "quality.jsonl").read_text().splitlines()
    ]
    rejected = next(row for row in quality if "markdown_line_break" in row["rejection_reasons"])
    assert receipt["accepted_count"] == 3
    assert "markdown_line_break" in rejected["rejection_reasons"]
    assert "unbalanced_quotation_marks" in rejected["rejection_reasons"]
    assert "banned_word_count=1" in rejected["rejection_reasons"]
    assert "stock_phrase_count=1" in rejected["rejection_reasons"]


def test_finalize_rejects_overlap_with_reference_corpus(tmp_path: Path) -> None:
    requests_path, _ = plan_generation(
        config_path(), request_count=1, output_directory=tmp_path / "plan"
    )
    examples = [candidate(index) for index in range(4)]
    generation_results = tmp_path / "generation.results.jsonl"
    write_jsonl(
        generation_results,
        [
            completion_result(
                "poetry-synthetic-00000000",
                {"examples": examples},
            )
        ],
    )
    candidates_path, critic_requests_path = ingest_generation_results(
        config_path(),
        requests_path=requests_path,
        results_path=generation_results,
        output_directory=tmp_path / "ingested",
    )
    critic_requests = [json.loads(line) for line in critic_requests_path.read_text().splitlines()]
    critic_results = tmp_path / "critic.results.jsonl"
    write_jsonl(
        critic_results,
        [
            completion_result(request["custom_id"], accepted_critique())
            for request in critic_requests
        ],
    )
    reference_poem = str(examples[0]["poem"])
    reference_document = SourceDocument(
        document_id="reference",
        provenance=Provenance(
            work="Reference",
            author="Test",
            licence="test",
            source="test fixture",
        ),
        text=reference_poem,
        blocks=(
            ContentBlock(
                block_id="reference:poem",
                kind="poem",
                text=reference_poem,
                poem_id="reference:poem",
            ),
        ),
    )
    reference_manifest = tmp_path / "reference.jsonl"
    write_manifest(reference_manifest, (reference_document,))

    receipt_path = finalize_synthetic_corpus(
        config_path(),
        candidates_path=candidates_path,
        critic_results_path=critic_results,
        output_directory=tmp_path / "corpus",
        reference_manifest=reference_manifest,
    )

    receipt = json.loads(receipt_path.read_text())
    quality_path = tmp_path / "corpus" / "quality.jsonl"
    quality = [json.loads(line) for line in quality_path.read_text().splitlines()]
    assert receipt["accepted_count"] == 3
    assert any("reference_shared_12_gram" in row["rejection_reasons"] for row in quality)


def test_config_rejects_unknown_fields(tmp_path: Path) -> None:
    value = json.loads(config_path().read_text())
    value["unexpected"] = True
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly"):
        SyntheticCorpusConfig.load(path)
