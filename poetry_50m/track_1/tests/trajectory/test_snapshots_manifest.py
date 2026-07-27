from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest
import torch

from poetry50m.trajectory.manifest import RunManifest, SuccessLevel, TrajectoryExperimentManifest
from poetry50m.trajectory.snapshots import (
    assert_single_run_trajectory,
    load_weight_snapshot,
    save_weight_snapshot,
)
from poetry50m.trajectory.types import SNAPSHOT_FORMAT, SnapshotMetadata

from .conftest import make_snapshot


def test_weights_only_snapshot_round_trip_and_malformed_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = make_snapshot(checkpoint_id="s0", step=0, state_dict={"weight": torch.ones(2)})
    path = tmp_path / "weights.pt"
    fsync_targets: list[bool] = []

    def record_fsync(descriptor: int) -> None:
        fsync_targets.append(stat.S_ISDIR(Path(f"/dev/fd/{descriptor}").stat().st_mode))

    monkeypatch.setattr("poetry50m.trajectory._persistence.os.fsync", record_fsync)
    save_weight_snapshot(path, snapshot)
    assert fsync_targets == [False, True]
    assert not list(tmp_path.glob(".weights.pt.*.tmp"))
    loaded = load_weight_snapshot(path, expected=snapshot)
    assert loaded.metadata == snapshot.metadata
    torch.testing.assert_close(loaded.state_dict["weight"], snapshot.state_dict["weight"])

    malformed = tmp_path / "malformed.pt"
    torch.save({"format": SNAPSHOT_FORMAT, "metadata": {}, "state_dict": {}}, malformed)
    with pytest.raises((TypeError, ValueError)):
        load_weight_snapshot(malformed)


def test_snapshot_metadata_rejects_untrusted_field_shapes_and_values() -> None:
    metadata = make_snapshot(
        checkpoint_id="s0", step=0, state_dict={"weight": torch.ones(1)}
    ).metadata.to_mapping()
    malformed_values: tuple[tuple[str, object], ...] = (
        ("run_id", 1),
        ("step", True),
        ("tokens_seen", True),
        ("wall_seconds", float("nan")),
        ("wall_seconds", float("inf")),
        ("wall_seconds", -0.1),
    )
    for field_name, malformed_value in malformed_values:
        candidate = {**metadata, field_name: malformed_value}
        with pytest.raises((TypeError, ValueError)):
            SnapshotMetadata.from_mapping(candidate)
    for candidate in (
        {key: value for key, value in metadata.items() if key != "run_id"},
        {**metadata, "unexpected": "value"},
    ):
        with pytest.raises(ValueError, match="exactly"):
            SnapshotMetadata.from_mapping(candidate)


def test_run_manifest_round_trip_rejects_untrusted_json(tmp_path: Path) -> None:
    manifest = RunManifest(
        "r0",
        "init-a",
        "order-a",
        "decoder",
        "corpus",
        "tokenizer",
        "code",
        "model-hash",
        "train-config",
        False,
        ("s0",),
    )
    path = tmp_path / "manifest.json"
    manifest.save(path)
    assert RunManifest.load(path) == manifest

    baseline = json.loads(path.read_text())
    malformed_values: tuple[tuple[str, object], ...] = (
        ("run_id", 1),
        ("endpoint_sealed", 1),
        ("fit_checkpoint_ids", ["s0", 1]),
    )
    for field_name, malformed_value in malformed_values:
        path.write_text(json.dumps({**baseline, field_name: malformed_value}))
        with pytest.raises((TypeError, ValueError)):
            RunManifest.load(path)
    for candidate in (
        {key: value for key, value in baseline.items() if key != "run_id"},
        {**baseline, "unexpected": "value"},
    ):
        path.write_text(json.dumps(candidate))
        with pytest.raises(ValueError, match="exactly"):
            RunManifest.load(path)


def test_raw_cross_run_and_cross_structure_trajectory_is_rejected() -> None:
    first = make_snapshot(checkpoint_id="s0", step=0, state_dict={"weight": torch.ones(2)})
    other_run = make_snapshot(
        run_id="r1", checkpoint_id="s1", step=1, state_dict={"weight": torch.ones(2)}
    )
    with pytest.raises(ValueError, match="run IDs"):
        assert_single_run_trajectory((first, other_run))
    other_structure = make_snapshot(
        checkpoint_id="s1",
        step=1,
        state_dict={"weight": torch.ones(3)},
        architecture_signature="decoder-v2",
    )
    with pytest.raises(ValueError, match="shape|architecture"):
        assert_single_run_trajectory((first, other_structure))


def test_sealed_level_two_manifest_rejects_r2_checkpoint_leakage() -> None:
    reference = RunManifest(
        run_id="r0",
        initialization_id="init-a",
        data_order_id="order-a",
        architecture_signature="decoder-v1",
        corpus_signature="corpus-v1",
        tokenizer_hash="tokenizer",
        code_signature="code",
        model_config_hash="model-hash",
        training_config_hash="train-config",
        endpoint_sealed=False,
        fit_checkpoint_ids=("r0-s0", "r0-s1"),
    )
    target = RunManifest(
        run_id="r2",
        initialization_id="init-a",
        data_order_id="order-b",
        architecture_signature="decoder-v1",
        corpus_signature="corpus-v1",
        tokenizer_hash="tokenizer",
        code_signature="code",
        model_config_hash="model-hash",
        training_config_hash="train-config",
        endpoint_sealed=True,
    )
    experiment = TrajectoryExperimentManifest(SuccessLevel.TRANSFER, reference, target)
    valid = make_snapshot(
        checkpoint_id="r0-s0",
        step=0,
        state_dict={"weight": torch.ones(2)},
        tokenizer_hash="tokenizer",
        code_signature="code",
        training_config_hash="train-config",
    )
    leaked = make_snapshot(
        run_id="r2",
        checkpoint_id="r2-s0",
        step=0,
        state_dict={"weight": torch.ones(2)},
        data_order_id="order-b",
        tokenizer_hash="tokenizer",
        code_signature="code",
        training_config_hash="train-config",
    )
    assert experiment.validate_fit_sources((valid,)) == (valid,)
    with pytest.raises(ValueError, match="non-reference|leakage"):
        experiment.validate_fit_sources((leaked,))


def test_level_two_requires_an_unseen_seed_or_order_and_sealing() -> None:
    reference = RunManifest(
        "r0",
        "init-a",
        "order-a",
        "decoder",
        "corpus",
        "tokenizer",
        "code",
        "model-hash",
        "train-config",
        False,
        ("s0",),
    )
    same = RunManifest(
        "r2",
        "init-a",
        "order-a",
        "decoder",
        "corpus",
        "tokenizer",
        "code",
        "model-hash",
        "train-config",
        True,
    )
    with pytest.raises(ValueError, match="unseen"):
        TrajectoryExperimentManifest(SuccessLevel.TRANSFER, reference, same)
    unsealed = RunManifest(
        "r2",
        "init-b",
        "order-b",
        "decoder",
        "corpus",
        "tokenizer",
        "code",
        "model-hash",
        "train-config",
        False,
    )
    with pytest.raises(ValueError, match="sealed"):
        TrajectoryExperimentManifest(SuccessLevel.TRANSFER, reference, unsealed)


def test_raw_level_two_explicitly_rejects_cross_seed_application() -> None:
    reference = RunManifest(
        "r0",
        "init-a",
        "order-a",
        "decoder",
        "corpus",
        "tokenizer",
        "code",
        "model-hash",
        "train-config",
        False,
        ("s0",),
    )
    target = RunManifest(
        "r2",
        "init-a",
        "order-b",
        "decoder",
        "corpus",
        "tokenizer",
        "code",
        "model-hash",
        "train-config",
        True,
    )
    experiment = TrajectoryExperimentManifest(SuccessLevel.TRANSFER, reference, target)
    source = make_snapshot(
        checkpoint_id="s0",
        step=0,
        state_dict={"weight": torch.ones(2)},
        architecture_signature="decoder",
        corpus_signature="corpus",
        tokenizer_hash="tokenizer",
        code_signature="code",
        training_config_hash="train-config",
    )
    cross_seed = make_snapshot(
        run_id="r2",
        checkpoint_id="s1",
        step=1,
        state_dict={"weight": torch.ones(2)},
        initialization_id="init-b",
        data_order_id="order-b",
        architecture_signature="decoder",
        corpus_signature="corpus",
        tokenizer_hash="tokenizer",
        code_signature="code",
        training_config_hash="train-config",
    )
    with pytest.raises(ValueError, match="cross-seed"):
        experiment.validate_target_snapshot(cross_seed, source)
