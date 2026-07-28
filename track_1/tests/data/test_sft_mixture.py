from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tokenizers import Tokenizer

from poetry50m.config import file_hash
from poetry50m.data.sft_mixture import build_sft_mixture
from poetry50m.data.tokenizer import save_tokenizer, train_tokenizer


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def messages(user: str, assistant: str) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


def formatted_count(tokenizer: Tokenizer, conversation: list[dict[str, str]]) -> int:
    return 4 + sum(
        len(tokenizer.encode(message["content"], add_special_tokens=False).ids)
        for message in conversation
    )


def supervised_count(tokenizer: Tokenizer, conversation: list[dict[str, str]]) -> int:
    response = conversation[-1]["content"]
    return len(tokenizer.encode(response, add_special_tokens=False).ids) + 1


@pytest.fixture
def tokenizer_path(tmp_path: Path) -> Path:
    tokenizer = train_tokenizer(
        (
            "write a poem rain rests on stone",
            "held out question answer new useful response",
        )
    )
    path = tmp_path / "tokenizer.json"
    save_tokenizer(tokenizer, path)
    return path


def test_build_sft_mixture_protects_test_and_reaches_exact_budget(
    tmp_path: Path, tokenizer_path: Path
) -> None:
    revision = "1" * 40
    acquisition = tmp_path / "acquired"
    test_path = acquisition / "data/test.parquet"
    train_path = acquisition / "data/train.parquet"
    test_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [{"messages": messages("held out question", "held out answer"), "source": "test"}]
        ),
        test_path,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "messages": messages("duplicate question", "held out answer"),
                    "source": "duplicate",
                },
                {
                    "messages": messages("new question", "new useful answer"),
                    "source": "eligible",
                },
                {
                    "messages": messages("held out question", "different answer"),
                    "source": "duplicate-test-prompt",
                },
                {
                    "messages": [
                        {"role": "user", "content": "first repeated question"},
                        {"role": "assistant", "content": "repeated answer"},
                        {"role": "user", "content": "second repeated question"},
                        {"role": "assistant", "content": "repeated answer"},
                    ],
                    "source": "duplicate-within-conversation",
                },
            ]
        ),
        train_path,
    )
    artifacts = [
        {
            "split": "test",
            "path": "data/test.parquet",
            "sha256": file_hash(test_path),
            "size_bytes": test_path.stat().st_size,
        },
        {
            "split": "train",
            "path": "data/train.parquet",
            "sha256": file_hash(train_path),
            "size_bytes": train_path.stat().st_size,
        },
    ]
    source_config = tmp_path / "sft_sources.json"
    write_json(
        source_config,
        {
            "format_version": 1,
            "sources": [
                {
                    "source_id": "smol_smoltalk",
                    "repository": "owner/repository",
                    "revision": revision,
                    "licence": "apache-2.0",
                    "artifacts": artifacts,
                }
            ],
        },
    )
    write_json(
        acquisition / "receipt.json",
        {
            "format_version": 1,
            "config_sha256": file_hash(source_config),
            "revision": revision,
        },
    )

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    synthetic_messages = messages("write a poem", "rain rests on stone")
    synthetic_record = {
        "format_version": 1,
        "example_id": "synthetic-sft-v2-000000000",
        "messages": synthetic_messages,
        "prompt_spec": {},
        "token_counts": {
            "prompt": len(
                tokenizer.encode("write a poem", add_special_tokens=False).ids
            ),
            "response": len(
                tokenizer.encode("rain rests on stone", add_special_tokens=False).ids
            ),
            "formatted": formatted_count(tokenizer, synthetic_messages),
            "training_input": formatted_count(tokenizer, synthetic_messages) - 1,
            "supervised": supervised_count(tokenizer, synthetic_messages),
        },
        "provenance": {"kind": "synthetic"},
    }
    synthetic_directory = tmp_path / "synthetic"
    synthetic_directory.mkdir()
    dataset_path = synthetic_directory / "dataset.jsonl"
    write_json(dataset_path, synthetic_record)
    synthetic_receipt = synthetic_directory / "receipt.json"
    write_json(
        synthetic_receipt,
        {
            "dataset_filename": dataset_path.name,
            "dataset_sha256": file_hash(dataset_path),
        },
    )
    heldout = tmp_path / "heldout.jsonl"
    heldout.write_text("", encoding="utf-8")
    evaluation = tmp_path / "evaluation.json"
    write_json(evaluation, {"cases": []})
    eligible_messages = messages("new question", "new useful answer")
    target = formatted_count(tokenizer, synthetic_messages) + formatted_count(
        tokenizer, eligible_messages
    )

    receipt_path = build_sft_mixture(
        source_config_path=source_config,
        acquisition_directory=acquisition,
        synthetic_receipt_path=synthetic_receipt,
        tokenizer_path=tokenizer_path,
        heldout_paths=[heldout],
        evaluation_suite_path=evaluation,
        output_directory=tmp_path / "mixture",
        target_tokens=target,
        expected_tokenizer_sha256=file_hash(tokenizer_path),
    )

    receipt = json.loads(receipt_path.read_text())
    rows = [
        json.loads(line)
        for line in (receipt_path.parent / "dataset.jsonl").read_text().splitlines()
    ]
    assert receipt["actual_formatted_tokens"] == target
    assert receipt["synthetic_example_count"] == 1
    assert receipt["smoltalk_example_count"] == 1
    assert receipt["rejection_counts"]["protected_lexical_family"] == 2
    assert receipt["rejection_counts"]["duplicate_assistant_within_conversation"] == 1
    assert receipt["protection_inputs"] == [
        {
            "kind": "track1_heldout",
            "path": str(heldout),
            "sha256": file_hash(heldout),
        },
        {
            "kind": "evaluation_suite",
            "path": str(evaluation),
            "sha256": file_hash(evaluation),
        },
    ]
    assert len(rows) == 2
    assert rows[1]["provenance"]["source"] == "eligible"
    assert file_hash(receipt_path.parent / "packs.jsonl") == receipt["packs_sha256"]
