from __future__ import annotations

import json
from pathlib import Path

from genome.hashing import sha256_file
from genome.io import load_json
from genome.sampling import partition_probe_sample, verify_independent_evaluation_sample


def _write_records(path: Path, values: list[list[int]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for input_ids in values:
            handle.write(json.dumps({"input_ids": input_ids}) + "\n")


def test_partition_probe_sample_is_disjoint_and_receipted(tmp_path: Path) -> None:
    source = tmp_path / "tokens.jsonl"
    records = [[index, index + 1] for index in range(6)]
    _write_records(source, records)

    receipt = partition_probe_sample(
        source_jsonl=source,
        output=tmp_path / "probes",
    )

    refinement = (tmp_path / "probes" / "refinement.jsonl").read_text().splitlines()
    evaluation = (tmp_path / "probes" / "evaluation.jsonl").read_text().splitlines()
    assert [json.loads(line)["input_ids"] for line in refinement] == records[::2]
    assert [json.loads(line)["input_ids"] for line in evaluation] == records[1::2]
    assert receipt["refinement"]["records"] == 3
    assert receipt["evaluation"]["records"] == 3
    assert receipt["refinement"]["sha256"] == sha256_file(tmp_path / "probes" / "refinement.jsonl")
    assert load_json(tmp_path / "probes" / "receipt.json") == receipt


def test_independent_verifier_requires_another_range_and_128_batches(tmp_path: Path) -> None:
    formula_tokens = tmp_path / "formula.jsonl"
    verifier_tokens = tmp_path / "verifier.jsonl"
    _write_records(formula_tokens, [[1, 2]])
    _write_records(verifier_tokens, [[index, index + 1] for index in range(128)])
    common = {
        "format": "GENOME_DATASET_SAMPLE",
        "repository": "EleutherAI/pile",
        "resolved_commit": "a" * 40,
    }
    formula_receipt = tmp_path / "formula-receipt.json"
    verifier_receipt = tmp_path / "verifier-receipt.json"
    formula_receipt.write_text(
        json.dumps(
            {
                **common,
                "filename": "shard-0.bin",
                "byte_range": {"start": 0, "end": 99},
                "examples": 1,
                "tokens_file": {
                    "path": str(formula_tokens),
                    "sha256": sha256_file(formula_tokens),
                },
            }
        )
    )
    verifier_receipt.write_text(
        json.dumps(
            {
                **common,
                "filename": "shard-1.bin",
                "byte_range": {"start": 0, "end": 999},
                "examples": 128,
                "tokens_file": {
                    "path": str(verifier_tokens),
                    "sha256": sha256_file(verifier_tokens),
                },
            }
        )
    )

    report = verify_independent_evaluation_sample(
        formula_sample_receipt=formula_receipt,
        verifier_receipt=verifier_receipt,
    )

    assert report["different_shard"] is True
    assert report["batches"] == 128
    assert report["evaluation_jsonl_sha256"] == sha256_file(verifier_tokens)
