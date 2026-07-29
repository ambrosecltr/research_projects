from __future__ import annotations

import pytest

from genome.source_audit import Decision, SourceAuditManifest, SourceCandidate, SourceSize, SplitPlan


def candidate(
    source_id: str,
    life_id: str,
    *,
    decision: Decision = "approved",
) -> SourceCandidate:
    approved = decision == "approved"
    return SourceCandidate(
        source_id=source_id,
        organization="research",
        repository_pattern=f"org/{source_id}",
        architecture_family="gpt_neox",
        licence="Apache-2.0",
        completeness="complete",
        decision=decision,
        priority=1,
        w0_status="verified_step0",
        final_endpoint_available=True,
        dataset_content_available=True,
        exact_data_order_available=True,
        tokenizer_available=True,
        complete_recipe_available=True,
        provenance_available=True,
        intermediate_checkpoints_available=True,
        sizes=(
            SourceSize(
                label="small",
                parameter_count=100,
                life_ids=(life_id,),
                checkpoint_count_per_life=2,
            ),
        ),
        approved_materialization=("W0", "WT") if approved else (),
        blocked_by=() if approved else ("evaluation only",),
        source_urls=(f"https://example.test/{source_id}",),
        notes=(),
    )


def test_complete_source_requires_all_training_and_provenance_inputs() -> None:
    value = candidate("public", "public-life").to_dict()
    for field in (
        "final_endpoint_available",
        "dataset_content_available",
        "exact_data_order_available",
        "tokenizer_available",
        "complete_recipe_available",
        "provenance_available",
    ):
        broken = dict(value)
        broken[field] = False
        with pytest.raises(ValueError, match=field):
            SourceCandidate.from_dict(broken)


def test_track1_evaluation_source_cannot_enter_active_compiler_split() -> None:
    track1 = candidate("track1", "track1-poetry", decision="evaluation_only")
    with pytest.raises(ValueError, match="non-approved"):
        SourceAuditManifest(
            candidates=(track1,),
            split_plan=SplitPlan(training=("track1-poetry",), development=(), hidden=()),
            assumptions={"storage_values": "estimates"},
        )


def test_revealed_round_one_seed9_must_remain_quarantined() -> None:
    with pytest.raises(ValueError, match="quarantined"):
        SplitPlan(training=(), development=(), hidden=("pythia-14m-seed9",))
    split = SplitPlan(
        training=(),
        development=(),
        hidden=(),
        quarantined=("pythia-14m-seed9",),
    )
    assert split.quarantined == ("pythia-14m-seed9",)
    revealed = candidate("round-one", "pythia-14m-seed9")
    with pytest.raises(ValueError, match="listed in the quarantine split"):
        SourceAuditManifest(
            candidates=(revealed,),
            split_plan=SplitPlan(training=(), development=(), hidden=()),
            assumptions={"storage_values": "estimates"},
        )
    manifest = SourceAuditManifest(
        candidates=(revealed,),
        split_plan=split,
        assumptions={"storage_values": "estimates"},
    )
    assert manifest.split_plan.quarantined == ("pythia-14m-seed9",)


def test_storage_math_is_explicitly_labelled_as_estimated() -> None:
    public = candidate("public", "public-life")
    audit = SourceAuditManifest(
        candidates=(public,),
        split_plan=SplitPlan(training=("public-life",), development=(), hidden=()),
        assumptions={"storage_values": "estimates_until_download_receipts"},
    )
    assert audit.estimated_approved_endpoint_pair_bytes() == 800
    assert audit.estimated_maximum_catalog_bytes() == 800
