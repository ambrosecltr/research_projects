from __future__ import annotations

import pytest

from genome.life_schema import (
    ArtifactRef,
    CheckpointRef,
    DatasetRef,
    ModelLifeManifest,
    TokenizerRef,
    TrainingStage,
    split_commitment,
    validate_life_splits,
)


def artifact(name: str, fill: str = "a") -> ArtifactRef:
    return ArtifactRef(
        uri=f"artifacts/{name}",
        sha256=fill * 64,
        bytes=123,
        revision="revision-1",
        licence="Apache-2.0",
    )


def complete_life(
    run_id: str = "life-a",
    *,
    lineage_id: str = "lineage-a",
    split: str = "training",
) -> ModelLifeManifest:
    hidden = split == "hidden"
    w0 = CheckpointRef("w0", 0, 0, "available", artifact(f"{run_id}-w0", "b"))
    pretrain = CheckpointRef(
        "pretrain-final",
        100,
        1_000_000,
        "available",
        artifact(f"{run_id}-pretrain", "c"),
    )
    endpoint = CheckpointRef(
        "wt",
        120,
        1_200_000,
        "sealed" if hidden else "available",
        None if hidden else artifact(f"{run_id}-wt", "d"),
    )
    dataset = DatasetRef(
        dataset_id="corpus",
        repository="org/corpus",
        revision="dataset-revision",
        configuration="default",
        split="train",
        licence="Apache-2.0",
        order_id="order-7",
        mixture_weight=1.0,
        semantic_fingerprint=artifact(f"{run_id}-fingerprint", "e"),
    )
    tokenizer = TokenizerRef(
        repository="org/tokenizer",
        revision="tokenizer-revision",
        tokenizer_class="BPE",
        vocab_size=256,
        special_tokens={"eos": 1},
        files=(artifact(f"{run_id}-tokenizer", "f"),),
    )
    stages = (
        TrainingStage(
            stage_id="pretraining",
            stage_type="pretraining",
            objective="next_token_prediction",
            dataset_ids=("corpus",),
            start_checkpoint_id="w0",
            end_checkpoint_id="pretrain-final",
            steps=100,
            tokens=1_000_000,
            context_length=128,
            global_batch_tokens=4096,
            data_order_id="order-7",
            precision="bf16",
            optimizer={"type": "AdamW", "learning_rate": 3e-4},
            schedule={"type": "cosine", "warmup_steps": 10},
        ),
        TrainingStage(
            stage_id="sft",
            stage_type="sft",
            objective="conditional_generation",
            dataset_ids=("corpus",),
            start_checkpoint_id="pretrain-final",
            end_checkpoint_id="wt",
            steps=20,
            tokens=200_000,
            context_length=128,
            global_batch_tokens=2048,
            data_order_id="sft-order",
            precision="bf16",
            optimizer={"type": "AdamW", "learning_rate": 1e-4},
            schedule={"type": "constant"},
        ),
    )
    return ModelLifeManifest(
        run_id=run_id,
        lineage_id=lineage_id,
        split=split,  # type: ignore[arg-type]
        completeness="complete",
        architecture_family="decoder_only_transformer",
        architecture={"hidden_size": 64, "layers": 2, "heads": 4},
        tensor_inventory=artifact(f"{run_id}-inventory", "1"),
        tokenizer=tokenizer,
        initialization=w0,
        datasets=(dataset,),
        stages=stages,
        trajectory=(pretrain,),
        endpoint=endpoint,
        compiler_evidence=artifact(f"{run_id}-evidence", "2"),
        fitted_program=None if hidden else artifact(f"{run_id}-program", "3"),
        evaluations={},
        source={"repository": "org/model", "revision": "model-revision"},
    )


def test_complete_multistage_life_round_trips() -> None:
    life = complete_life()
    restored = ModelLifeManifest.from_dict(life.to_dict())
    assert restored == life
    assert restored.content_sha256 == life.content_sha256
    assert [stage.stage_type for stage in restored.stages] == ["pretraining", "sft"]


def test_compiler_view_excludes_endpoint_and_provenance_hashes() -> None:
    view = complete_life().compiler_view()
    encoded = repr(view)
    assert "endpoint" not in view
    assert "trajectory" not in view
    assert "fitted_program" not in view
    assert "run_id" not in view
    assert "sha256" not in encoded
    assert "order-7" not in encoded
    assert "sft-order" not in encoded
    assert "org/model" not in encoded
    assert "model-revision" not in encoded
    assert [stage["stage_type"] for stage in view["stages"]] == ["pretraining", "sft"]


def test_hidden_complete_life_requires_sealed_endpoint() -> None:
    hidden = complete_life("hidden-life", lineage_id="hidden-lineage", split="hidden")
    assert hidden.endpoint is not None
    assert hidden.endpoint.access == "sealed"
    assert hidden.fitted_program is None

    value = hidden.to_dict()
    value["endpoint"] = CheckpointRef(
        "wt",
        120,
        1_200_000,
        "available",
        artifact("leaked-wt", "9"),
    ).to_dict()
    with pytest.raises(ValueError, match="must not expose WT"):
        ModelLifeManifest.from_dict(value)


def test_stage_chain_must_start_at_true_w0_and_end_at_wt() -> None:
    value = complete_life().to_dict()
    value["stages"][0]["start_checkpoint_id"] = "pretrained"
    with pytest.raises(ValueError, match="true initialization"):
        ModelLifeManifest.from_dict(value)


def test_split_commitment_rejects_lineage_leakage() -> None:
    training = complete_life("train", lineage_id="same", split="training")
    hidden = complete_life("hidden", lineage_id="same", split="hidden")
    with pytest.raises(ValueError, match="spans"):
        validate_life_splits((training, hidden))

    development = complete_life("dev", lineage_id="dev-lineage", split="development")
    commitment = split_commitment((training, development))
    assert commitment["splits"] == {
        "training": ["train"],
        "development": ["dev"],
        "hidden": [],
    }
    assert len(commitment["content_sha256"]) == 64
