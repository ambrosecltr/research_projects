from pathlib import Path

import pytest

from genome.protocol import ArtifactBinding, TargetFormula


def test_checked_in_formula_id_matches_immutable_formula() -> None:
    formula = TargetFormula.load(
        Path(__file__).parents[1] / "configs" / "targets" / "pythia_v1.yaml"
    )

    assert (
        formula.formula_id
        == "4f4e6d9d5d9ef7677dd955bb89be81dfedf161ecb010fdfd405475fdce46d155"
    )
    assert formula.status == "formula-development"
    assert formula.data["development_evaluation_batches"] == 128


def test_artifact_binding_requires_full_digests() -> None:
    values = {
        "run_id": "pythia-14m-seed0",
        "formula_id": "a" * 64,
        "program_id": "b" * 64,
        "program_manifest_sha256": "c" * 64,
        "payload_sha256": "d" * 64,
        "w0_state_id": "e" * 64,
        "wt_state_id": "f" * 64,
        "evaluation_jsonl_sha256": "0" * 64,
        "source_plan_id": "1" * 64,
        "code_commit": "2" * 40,
    }
    assert ArtifactBinding.from_dict(values).run_id == "pythia-14m-seed0"

    with pytest.raises(ValueError, match="formula_id"):
        ArtifactBinding.from_dict({**values, "formula_id": "short"})
