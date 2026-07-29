from __future__ import annotations

import pytest

from genome.sources import default_pythia_v1_plan, materialize_plan


def test_default_pythia_plan_is_fresh_and_whole_life_split() -> None:
    plan = default_pythia_v1_plan()
    assert len(plan.lives) == 19
    assert sum(item.split == "training" for item in plan.lives) == 16
    assert sum(item.split == "development" for item in plan.lives) == 2
    hidden = [item for item in plan.lives if item.split == "hidden"]
    assert [(item.size, item.seed) for item in hidden] == [("31m", 9)]
    assert all(not (item.size == "14m" and item.seed == 9) for item in plan.lives)
    assert hidden[0].wt_commit is None


def test_unresolved_plan_cannot_download(tmp_path) -> None:
    with pytest.raises(ValueError, match="resolved and pinned"):
        materialize_plan(default_pythia_v1_plan(), root=tmp_path)
