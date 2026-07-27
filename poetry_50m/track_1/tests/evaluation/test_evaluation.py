from __future__ import annotations

from pathlib import Path

import pytest

from poetry50m.data.schema import TokenSequence
from poetry50m.evaluation.metrics import (
    heldout_loss_inputs,
    keyword_relevance,
    repetition_metrics,
    structural_metrics,
    training_overlap,
)
from poetry50m.evaluation.schema import (
    BlindComparison,
    BlindComparisonPack,
    BlindJudgment,
    CostRecord,
    CriterionTally,
    GenerationRequest,
    PromptCase,
    PromptSuite,
    aggregate_blind_judgments,
    blind_comparison_pack,
    generation_requests,
    multi_seed_generation_requests,
)


def suite() -> PromptSuite:
    return PromptSuite(
        "fixture-suite",
        1,
        (
            PromptCase("river", "Write beside a river.", ("river", "stone"), 2),
            PromptCase("dawn", "Write at dawn.", ("dawn", "bird"), 2),
        ),
    )


def test_generation_manifest_and_blind_pack_are_reproducible():
    requests = multi_seed_generation_requests(
        suite(),
        checkpoint_id="step-20",
        seeds=(1, 2, 3),
        max_new_tokens=40,
        temperature=0.8,
        top_p=0.9,
    )
    other_checkpoint = multi_seed_generation_requests(
        suite(),
        checkpoint_id="step-80",
        seeds=(1, 2, 3),
        max_new_tokens=40,
        temperature=0.8,
        top_p=0.9,
    )
    assert [request.request_id for request in requests] == [
        request.request_id for request in other_checkpoint
    ]
    changed_settings = generation_requests(
        suite(),
        checkpoint_id="step-20",
        seed=1,
        max_new_tokens=41,
        temperature=0.8,
        top_p=0.9,
    )
    assert requests[0].request_id != changed_settings[0].request_id
    assert (
        multi_seed_generation_requests(
            suite(),
            checkpoint_id="step-20",
            seeds=(1, 2, 3),
            max_new_tokens=40,
            temperature=0.8,
            top_p=0.9,
        )
        == requests
    )
    pairs = blind_comparison_pack(
        requests=requests,
        outputs_a={request.request_id: f"A {request.seed}" for request in requests},
        outputs_b={request.request_id: f"B {request.seed}" for request in requests},
        blind_seed=2,
        candidate_a_id="baseline",
        candidate_b_id="transport",
    )
    assert len(pairs.comparisons) == 6
    assert [pair.left_label for pair in pairs.comparisons] == ["A"] * 6
    assert {pair.case_id for pair in pairs.comparisons} == {"river", "dawn"}
    assert {pair.seed for pair in pairs.comparisons} == {1, 2, 3}
    assert set(pairs.unblinding_key) == {pair.comparison_id for pair in pairs.comparisons}
    judgments = [
        BlindJudgment(pair.comparison_id, "A", "B", "tie", "A") for pair in pairs.comparisons
    ]
    tallies = aggregate_blind_judgments(pairs, judgments)
    assert {tally.criterion for tally in tallies} == {
        "prompt_relevance",
        "poetic_quality",
        "image_music",
        "degeneration",
    }
    relevance = next(tally for tally in tallies if tally.criterion == "prompt_relevance")
    expected_baseline_wins = sum(
        mapping["A"] == "baseline" for mapping in pairs.unblinding_key.values()
    )
    assert relevance.candidate_a_wins == expected_baseline_wins
    assert (
        len(
            multi_seed_generation_requests(
                suite(),
                checkpoint_id="step-20",
                seeds=(1, 2, 3),
                max_new_tokens=40,
                temperature=0.8,
                top_p=0.9,
            )
        )
        == 6
    )


def test_blind_pack_rejects_malformed_keys_and_duplicate_judgments():
    comparison = BlindComparison(
        "comparison",
        "request",
        "case",
        1,
        "A",
        "B",
        "left poem",
        "right poem",
    )
    valid_key = {"comparison": {"A": "baseline", "B": "candidate"}}
    pack = BlindComparisonPack((comparison,), valid_key, "baseline", "candidate")
    with pytest.raises(ValueError, match="exactly cover"):
        BlindComparisonPack((comparison,), {}, "baseline", "candidate")
    with pytest.raises(ValueError, match="exactly A and B"):
        BlindComparisonPack(
            (comparison,),
            {"comparison": {"A": "baseline", "C": "candidate"}},
            "baseline",
            "candidate",
        )
    with pytest.raises(ValueError, match="bijectively"):
        BlindComparisonPack(
            (comparison,),
            {"comparison": {"A": "baseline", "B": "baseline"}},
            "baseline",
            "candidate",
        )
    with pytest.raises(ValueError, match="unique"):
        BlindComparisonPack((comparison, comparison), valid_key, "baseline", "candidate")
    with pytest.raises(ValueError, match="exactly A and B"):
        BlindComparison("bad", "request", "case", 1, "left", "right", "one", "two")

    judgment = BlindJudgment("comparison", "A", "B", "tie", "A")
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_blind_judgments(pack, (judgment, judgment))


def test_fixed_suite_has_forty_evaluation_cases_and_ten_development_cases():
    path = Path(__file__).parents[2] / "configs" / "evaluation" / "prompt_suite.json"
    fixed = PromptSuite.load(path)
    assert sum(case.partition == "evaluation" for case in fixed.cases) == 40
    assert sum(case.partition == "development" for case in fixed.cases) == 10


def test_overlap_degeneracy_structure_relevance_and_heldout_inputs():
    overlap = training_overlap("The river waits in rain", ["The river waits in rain."])
    assert overlap.exact_match and overlap.maximum_ngram >= 4
    across_boundary = training_overlap("river waits", ["river", "waits"])
    assert not across_boundary.exact_match and across_boundary.maximum_ngram == 1
    degeneration = repetition_metrics("a river\na river\n\na river")
    assert degeneration.repeated_line_rate > 0
    structure = structural_metrics("one line\ntwo line\n\nthird line")
    assert structure.stanza_count == 2 and structure.line_count == 3
    assert keyword_relevance("A river holds a stone", ("river", "stone", "dawn")) == 2 / 3
    heldout = heldout_loss_inputs([TokenSequence("x", "x", (1, 2, 3), (False, True, True))])
    assert heldout[0].target_positions == (1, 2)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_cost_record_rejects_non_finite_values(value: float):
    with pytest.raises(ValueError, match="finite"):
        CostRecord("run", "checkpoint", 1, 1, value, 1.0)


def test_cost_record_preserves_unknown_accelerator_time() -> None:
    record = CostRecord(
        "run",
        "checkpoint",
        1,
        1,
        2.0,
        None,
        device_active_wall_seconds=1.5,
    )
    assert record.accelerator_seconds is None
    assert record.device_active_wall_seconds == 1.5
    with pytest.raises(ValueError, match="finite"):
        CostRecord(
            "run",
            "checkpoint",
            1,
            1,
            2.0,
            None,
            device_active_wall_seconds=float("nan"),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("suite_version", True),
        ("seed", False),
        ("max_new_tokens", True),
        ("temperature", float("nan")),
        ("temperature", float("inf")),
        ("top_p", float("nan")),
    ],
)
def test_generation_request_rejects_invalid_identity_numbers(field: str, value: object):
    arguments: dict[str, object] = {
        "request_id": "request",
        "suite_id": "suite",
        "suite_version": 1,
        "case_id": "case",
        "prompt": "prompt",
        "checkpoint_id": "checkpoint",
        "seed": 1,
        "max_new_tokens": 10,
        "temperature": 0.8,
        "top_p": 0.9,
    }
    arguments[field] = value
    with pytest.raises(ValueError):
        GenerationRequest(**arguments)


def test_evaluation_schema_rejects_unknown_keys_and_boolean_counts():
    valid_case = {
        "case_id": "case",
        "prompt": "Write a river.",
        "keywords": ["river"],
    }
    with pytest.raises(ValueError, match="unknown prompt suite"):
        PromptSuite.from_mapping(
            {"suite_id": "suite", "version": 1, "cases": [valid_case], "typo": True}
        )
    with pytest.raises(ValueError, match="unknown prompt case"):
        PromptSuite.from_mapping(
            {
                "suite_id": "suite",
                "version": 1,
                "cases": [{**valid_case, "typo": True}],
            }
        )
    with pytest.raises(ValueError, match="stanza"):
        PromptCase("case", "prompt", ("river",), expected_stanza_count=True)
    with pytest.raises(ValueError, match="seed"):
        BlindComparison("comparison", "request", "case", True, "A", "B", "one", "two")
    with pytest.raises(ValueError, match="tallies"):
        CriterionTally("prompt_relevance", True, 0, 0)
    with pytest.raises(ValueError, match="counts"):
        CostRecord("run", "checkpoint", True, 1, 1.0, 1.0)
