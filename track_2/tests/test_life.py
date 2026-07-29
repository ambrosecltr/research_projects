from __future__ import annotations

from dataclasses import replace

import pytest

from genome.life import (
    ArtifactRef,
    CheckpointRef,
    DatasetRef,
    LifeSplits,
    ModelLife,
    TokenizerRef,
    TrainingStage,
)

SHA = "a" * 64


def artifact(uri: str) -> ArtifactRef:
    return ArtifactRef(uri=uri, sha256=SHA, bytes=1, revision="commit", licence="Apache-2.0")


def checkpoint(name: str, step: int, tokens: int, access: str = "available") -> CheckpointRef:
    return CheckpointRef(
        checkpoint_id=name,
        step=step,
        tokens_seen=tokens,
        access=access,
        artifact=artifact(name) if access == "available" else None,
    )


def life(split: str = "training") -> ModelLife:
    w0 = checkpoint("w0", 0, 0)
    mid = checkpoint("mid", 10, 100)
    wt = checkpoint("wt", 20, 200, "sealed" if split == "hidden" else "available")
    return ModelLife(
        run_id=f"run-{split}",
        lineage_id=f"lineage-{split}",
        split=split,
        completeness="complete",
        architecture_family="gpt_neox",
        architecture={"hidden_size": 8},
        tensor_inventory=artifact("inventory"),
        tokenizer=TokenizerRef(
            repository="repo",
            revision="commit",
            tokenizer_class="Tokenizer",
            vocab_size=32,
            special_tokens={"eos": 0},
            files=(artifact("tokenizer"),),
        ),
        initialization=w0,
        datasets=(
            DatasetRef(
                dataset_id="pile",
                repository="dataset",
                revision="commit",
                split="train",
                licence="MIT",
                order_id="order",
                mixture_weight=1.0,
                semantic_evidence=artifact("fingerprint"),
            ),
        ),
        stages=(
            TrainingStage(
                stage_id="pretrain",
                kind="pretraining",
                objective="causal_lm",
                dataset_ids=("pile",),
                start_checkpoint_id="w0",
                end_checkpoint_id="wt" if split == "hidden" else "mid",
                steps=20 if split == "hidden" else 10,
                tokens=200 if split == "hidden" else 100,
                context_length=16,
                global_batch_tokens=32,
                data_order_id="order",
                precision="bf16",
                optimizer={"name": "adamw"},
                schedule={"name": "cosine"},
            ),
        ) if split == "hidden" else (
            TrainingStage(
                stage_id="pretrain-a",
                kind="pretraining",
                objective="causal_lm",
                dataset_ids=("pile",),
                start_checkpoint_id="w0",
                end_checkpoint_id="mid",
                steps=10,
                tokens=100,
                context_length=16,
                global_batch_tokens=32,
                data_order_id="order",
                precision="bf16",
                optimizer={"name": "adamw"},
                schedule={"name": "cosine"},
            ),
            TrainingStage(
                stage_id="pretrain-b",
                kind="continued_pretraining",
                objective="causal_lm",
                dataset_ids=("pile",),
                start_checkpoint_id="mid",
                end_checkpoint_id="wt",
                steps=10,
                tokens=100,
                context_length=16,
                global_batch_tokens=32,
                data_order_id="order",
                precision="bf16",
                optimizer={"name": "adamw"},
                schedule={"name": "cosine"},
            ),
        ),
        trajectory=() if split == "hidden" else (mid,),
        endpoint=wt,
        compiler_evidence=artifact("evidence"),
    )


def test_complete_life_and_hidden_view() -> None:
    training = life()
    assert training.compiler_view()["architecture_family"] == "gpt_neox"
    hidden = life("hidden")
    view = hidden.compiler_view()
    assert "endpoint" not in view
    assert "trajectory" not in view
    assert "sha256" not in str(view)


def test_stage_checkpoint_must_be_declared() -> None:
    item = life()
    bad_stage = replace(item.stages[0], end_checkpoint_id="invented")
    with pytest.raises(ValueError, match="undeclared"):
        replace(item, stages=(bad_stage, item.stages[1]))


def test_checkpoint_order_is_monotonic() -> None:
    item = life()
    later = checkpoint("later", 30, 300)
    earlier = checkpoint("earlier", 5, 50)
    with pytest.raises(ValueError, match="monotonic"):
        replace(item, trajectory=(later, earlier))


def test_whole_life_splits_are_disjoint() -> None:
    split = LifeSplits(training=("a",), development=("b",), hidden=("c",))
    assert split.assignment("c") == "hidden"
    with pytest.raises(ValueError, match="only one split"):
        LifeSplits(training=("a",), development=("a",), hidden=("c",))
