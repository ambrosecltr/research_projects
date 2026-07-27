from __future__ import annotations

import json
import shutil

import pytest
import torch

from genome.evaluator import evaluate_model_state
from genome.specimen import load_specimen, verify_specimen_files
from genome.tensor_inventory import assert_tied_equal


def test_specimen_integrity_and_reproducibility(tiny_artifacts):
    specimen = load_specimen(tiny_artifacts["specimen"].root)
    assert verify_specimen_files(specimen)["valid"]
    base = specimen.load_base()
    target = specimen.load_target()
    assert set(base) == set(target) == {spec.name for spec in specimen.inventory}
    assert_tied_equal(base, specimen.tied_groups)
    assert_tied_equal(target, specimen.tied_groups)
    assert any(not torch.equal(base[name], target[name]) for name in base)
    assert specimen.manifest["base_reproducibility"]["verified"] is True
    assert specimen.manifest["base_validation"]["valid_base"] is True
    assert specimen.manifest["endpoint_validation"]["complete"] is True


def test_r0_is_better_than_w0(tiny_artifacts):
    specimen = tiny_artifacts["specimen"]
    adapter = tiny_artifacts["adapter"]
    w0 = evaluate_model_state(adapter, specimen.load_base(), split="development", max_batches=2)
    wt = evaluate_model_state(adapter, specimen.load_target(), split="development", max_batches=2)
    assert wt["mean_loss"] < w0["mean_loss"]


def test_function_evaluation_retains_only_compact_anchor_logits(tiny_artifacts):
    specimen = tiny_artifacts["specimen"]
    adapter = tiny_artifacts["adapter"]
    metrics = evaluate_model_state(
        adapter,
        specimen.load_target(),
        split="development",
        max_batches=2,
        capture_logits=True,
        anchor_positions_per_batch=3,
    )
    assert len(metrics["logits"]) == 2
    assert all(part.ndim == 2 and 1 <= part.shape[0] <= 3 for part in metrics["logits"])


def test_specimen_rejects_path_traversal(tiny_artifacts):
    source = tiny_artifacts["specimen"].root
    escaped = tiny_artifacts["root"] / "escaped_specimen"
    shutil.copytree(source, escaped)
    manifest_path = escaped / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["tensor_inventory"] = "../tensor_inventory.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="files.tensor_inventory"):
        load_specimen(escaped)
