from __future__ import annotations

import pytest

from genome.io import load_json
from genome.sources import SourcePlan, default_pythia_v1_plan, materialize_plan


def test_default_pythia_plan_is_fresh_and_whole_life_split() -> None:
    plan = default_pythia_v1_plan()
    assert len(plan.lives) == 20
    assert sum(item.split == "training" for item in plan.lives) == 17
    assert sum(item.split == "development" for item in plan.lives) == 2
    hidden = [item for item in plan.lives if item.split == "hidden"]
    assert [(item.size, item.seed) for item in hidden] == [("31m", 9)]
    seed9_14m = [
        item for item in plan.lives if item.size == "14m" and item.seed == 9
    ]
    assert len(seed9_14m) == 1
    assert seed9_14m[0].split == "training"
    assert seed9_14m[0].w0_revision == "step0"
    assert seed9_14m[0].wt_revision == "step143000"
    assert hidden[0].wt_commit is None


def test_unresolved_plan_cannot_download(tmp_path) -> None:
    with pytest.raises(ValueError, match="resolved and pinned"):
        materialize_plan(default_pythia_v1_plan(), root=tmp_path)


def test_checked_in_source_and_compiler_corpus_plans_match_v1_split() -> None:
    assert SourcePlan.load("configs/sources/pythia_v1.json") == default_pythia_v1_plan()
    corpus_template = load_json("configs/corpus/compiler_v1.template.json")
    assert corpus_template["expected_records"] == {
        "training": 17,
        "development": 2,
        "total": 19,
    }
