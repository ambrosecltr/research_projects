from __future__ import annotations

import json

import pytest

from genome.evaluator import GenomeGate
from genome.rate_distortion import RateDistortionPoint, run_rate_distortion


def test_rate_distortion_reuses_svd_and_is_immutable(tiny_artifacts):
    specimen = tiny_artifacts["specimen"]
    gate = GenomeGate(
        tiny_artifacts["adapter"],
        specimen,
        split="development",
        max_batches=1,
    )
    output = tiny_artifacts["root"] / "rate_distortion"
    points = [
        RateDistortionPoint("svd", "rank1", rank=1),
        RateDistortionPoint("svd", "rank2", rank=2),
    ]
    results = run_rate_distortion(specimen, gate, output_dir=output, points=points)
    assert len(results) == 2
    context = json.loads((output / "rate_distortion_context.json").read_text())
    shared = context["shared_svd_workspace"]
    assert shared["built"] is True
    assert shared["matrix_count"] > 0
    assert shared["reused_by_candidates"] == ["rank1", "rank2"]
    assert shared["accounting_policy"] == "charge_once_across_the_frontier"
    assert all(
        result["compute"]["shared_svd_cost_policy"]
        == "charge_once_across_the_frontier"
        for result in results
    )
    with pytest.raises(FileExistsError, match="already exists"):
        run_rate_distortion(specimen, gate, output_dir=output, points=points)
