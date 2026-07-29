from __future__ import annotations

import json
from pathlib import Path

from genome.hashing import sha256_file
from genome.io import load_json
from genome.sampling import partition_probe_sample


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
