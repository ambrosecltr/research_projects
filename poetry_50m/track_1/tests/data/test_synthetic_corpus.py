from __future__ import annotations

import json
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
    finalize_synthetic_corpus,
    ingest_generation_results,
    merge_corpus_artifacts,
    plan_generation,
)


def config_path() -> Path:
    return Path(__file__).parents[2] / "configs" / "data" / "synthetic_cerebras_v1.json"


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
        request["body"]["response_format"]["json_schema"]["strict"] is True
        for request in requests
    )
    assert "minItems" not in json.dumps(requests)
    assert "maxItems" not in json.dumps(requests)
    assert json.loads(plan_path.read_text())["candidate_capacity"] == 16


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
    critic_requests = [
        json.loads(line) for line in critic_requests_path.read_text().splitlines()
    ]
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
    documents = tuple(
        iter_manifest(tmp_path / "corpus" / "manifest.jsonl", allow_synthetic=True)
    )
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
    critic_requests = [
        json.loads(line) for line in critic_requests_path.read_text().splitlines()
    ]
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
    quality = [
        json.loads(line) for line in quality_path.read_text().splitlines()
    ]
    assert receipt["accepted_count"] == 3
    assert any("repeated_lines" in row["rejection_reasons"] for row in quality)


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
    critic_requests = [
        json.loads(line) for line in critic_requests_path.read_text().splitlines()
    ]
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
    quality = [
        json.loads(line) for line in quality_path.read_text().splitlines()
    ]
    assert receipt["accepted_count"] == 3
    assert any(
        "reference_shared_12_gram" in row["rejection_reasons"] for row in quality
    )


def test_config_rejects_unknown_fields(tmp_path: Path) -> None:
    value = json.loads(config_path().read_text())
    value["unexpected"] = True
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly"):
        SyntheticCorpusConfig.load(path)
