from __future__ import annotations

import json
import os
import random
import stat
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from poetry50m.model import DecoderOnlyTransformer, ModelConfig
from poetry50m.training import CyclingBatchStream, TrainConfig, Trainer, mapping_hash
from poetry50m.training.engine import capture_rng_state, restore_rng_state, seed_everything
from poetry50m.trajectory.forecast import ForecastResult
from poetry50m.trajectory.gates import CandidateDecision, ForecastSafetyReport, FunctionDriftMetrics
from poetry50m.trajectory.preparation import prepare_forecast
from poetry50m.trajectory.snapshots import load_weight_snapshot
from poetry50m.trajectory.types import SnapshotMetadata
from poetry50m.trajectory.verification import CandidateVerification, VerificationTiming


class _UntrustedPayload:
    pass


def model() -> DecoderOnlyTransformer:
    return DecoderOnlyTransformer(
        ModelConfig(
            architecture="gpt",
            vocab_size=32,
            max_seq_len=8,
            d_model=16,
            n_layers=2,
            n_heads=4,
            ffn_dim=32,
        )
    )


def batches() -> list[dict[str, torch.Tensor | list[str] | int]]:
    torch.manual_seed(91)
    return [
        {
            "input_ids": torch.randint(0, 32, (2, 8)),
            "targets": torch.randint(0, 32, (2, 8)),
            "loss_mask": torch.ones((2, 8)),
            "example_ids": ["a", "b"],
            "data_token_count": 14,
        },
        {
            "input_ids": torch.randint(0, 32, (2, 8)),
            "targets": torch.randint(0, 32, (2, 8)),
            "loss_mask": torch.ones((2, 8)),
            "example_ids": ["c", "d"],
            "data_token_count": 15,
        },
    ]


def train_config(max_steps: int) -> TrainConfig:
    return TrainConfig(
        max_steps=max_steps,
        learning_rate=1e-3,
        weight_decay=0.0,
        device="cpu",
        precision="none",
        seed=123,
        checkpoint_every_steps=0,
        trajectory_every_steps=0,
    )


def trajectory_metadata(model_config: ModelConfig, train_config: TrainConfig) -> SnapshotMetadata:
    return SnapshotMetadata(
        run_id="run-1",
        checkpoint_id="initial",
        step=0,
        initialization_id="init-1",
        data_order_id="order-1",
        architecture_signature="gpt-tiny",
        corpus_signature="corpus-1",
        model_config_hash=mapping_hash(asdict(model_config)),
        tokenizer_hash="tokenizer-1",
        code_signature="code-1",
        training_config_hash=mapping_hash(asdict(train_config)),
    )


def test_tiny_train_loop_records_telemetry_and_per_example_losses(tmp_path: Path) -> None:
    observed: list[tuple[int, list[str], torch.Tensor]] = []

    def hook(step: int, identifiers: list[str] | None, losses: torch.Tensor) -> None:
        assert identifiers is not None
        observed.append((step, identifiers, losses.clone()))

    model_instance = model()
    config = train_config(2)
    trainer = Trainer(
        model_instance,
        config,
        tmp_path,
        per_example_loss_hook=hook,
        trajectory_metadata=trajectory_metadata(model_instance.config, config),
    )
    state = trainer.fit(CyclingBatchStream(batches()))
    assert state.global_step == 2
    assert state.data_tokens_seen == 29
    assert state.supervised_tokens_seen == 32
    assert len(observed) == 2
    assert len((tmp_path / "telemetry.jsonl").read_text().splitlines()) == 2
    snapshot = trainer.save_trajectory_snapshot()
    loaded = load_weight_snapshot(snapshot)
    assert loaded.metadata.tokens_seen == 29
    assert loaded.metadata.checkpoint_id == "step-00000002"
    assert (
        any(
            "layer_analysis" in line
            for line in (tmp_path / "telemetry.jsonl").read_text().splitlines()
        )
        is False
    )


def test_checkpoint_resume_is_bitwise_on_cpu(tmp_path: Path) -> None:
    torch.manual_seed(11)
    initial_model = model()
    initial_state = {
        key: value.detach().clone() for key, value in initial_model.state_dict().items()
    }

    uninterrupted_model = model()
    uninterrupted_model.load_state_dict(initial_state)
    uninterrupted = Trainer(uninterrupted_model, train_config(4), tmp_path / "uninterrupted")
    uninterrupted.fit(CyclingBatchStream(batches()))

    split_model = model()
    split_model.load_state_dict(initial_state)
    split = Trainer(split_model, train_config(4), tmp_path / "split")
    split_stream = CyclingBatchStream(batches())
    split.fit(split_stream, until_step=2)
    checkpoint = split.save_checkpoint()

    resumed_model = model()
    resumed = Trainer(resumed_model, train_config(4), tmp_path / "resumed")
    resumed_stream = CyclingBatchStream(batches())
    resumed.load_checkpoint(checkpoint, resumed_stream)
    resumed.fit(resumed_stream)

    assert resumed.state.global_step == 4
    assert resumed.state.data_tokens_seen == uninterrupted.state.data_tokens_seen == 58
    assert resumed.state.supervised_tokens_seen == uninterrupted.state.supervised_tokens_seen == 64
    for expected, actual in zip(
        uninterrupted.model.parameters(), resumed.model.parameters(), strict=True
    ):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)


def test_rng_state_uses_portable_numpy_words_and_restores_exactly(tmp_path: Path) -> None:
    seed_everything(29, deterministic=True)
    state = capture_rng_state()
    assert state["numpy"]["state"].dtype == torch.int64

    path = tmp_path / "rng.pt"
    torch.save(state, path)
    loaded = torch.load(path, weights_only=True)

    expected_python = random.random()
    expected_numpy = np.random.random()
    expected_torch = torch.rand(4)
    restore_rng_state(loaded)

    assert random.random() == expected_python
    assert np.random.random() == expected_numpy
    torch.testing.assert_close(torch.rand(4), expected_torch, rtol=0, atol=0)


def test_checkpoint_save_fsyncs_file_and_parent_and_loads_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = Trainer(model(), train_config(2), tmp_path / "source")
    stream = CyclingBatchStream(batches())
    source.fit(stream, until_step=1)
    synced_directory_flags: list[bool] = []
    real_fsync = os.fsync

    def tracked_fsync(descriptor: int) -> None:
        synced_directory_flags.append(stat.S_ISDIR(os.fstat(descriptor).st_mode))
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", tracked_fsync)
    checkpoint = source.save_checkpoint()
    assert synced_directory_flags == [False, True]

    resumed = Trainer(model(), train_config(2), tmp_path / "resumed")
    resumed_stream = CyclingBatchStream(batches())
    resumed.load_checkpoint(checkpoint, resumed_stream)
    assert asdict(resumed.state) == asdict(source.state)
    assert resumed_stream.state_dict() == stream.state_dict()
    for expected, actual in zip(source.model.parameters(), resumed.model.parameters(), strict=True):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)


def test_checkpoint_load_can_suppress_only_resume_telemetry(tmp_path: Path) -> None:
    source = Trainer(model(), train_config(2), tmp_path / "source")
    stream = CyclingBatchStream(batches())
    source.fit(stream, until_step=1)
    checkpoint = source.save_checkpoint()

    reader = Trainer(model(), train_config(2), tmp_path / "reader")
    reader.load_checkpoint(
        checkpoint,
        CyclingBatchStream(batches()),
        record_resume_event=False,
    )
    assert not (tmp_path / "reader" / "telemetry.jsonl").exists()

    resumer = Trainer(model(), train_config(2), tmp_path / "resumer")
    resumer.load_checkpoint(checkpoint, CyclingBatchStream(batches()))
    event = json.loads((tmp_path / "resumer" / "telemetry.jsonl").read_text())
    assert event["event"] == "resume"
    with pytest.raises(TypeError, match="boolean"):
        resumer.load_checkpoint(
            checkpoint,
            CyclingBatchStream(batches()),
            record_resume_event=1,
        )


def test_seed_everything_explicitly_toggles_deterministic_policy() -> None:
    previous_enabled = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    previous_benchmark = torch.backends.cudnn.benchmark
    previous_cudnn_deterministic = torch.backends.cudnn.deterministic
    previous_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    try:
        seed_everything(17, deterministic=True)
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
        assert torch.backends.cudnn.deterministic
        assert not torch.backends.cudnn.benchmark
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"

        seed_everything(17, deterministic=False)
        assert not torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
        assert not torch.backends.cudnn.deterministic
        assert torch.backends.cudnn.benchmark
        assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ
    finally:
        torch.use_deterministic_algorithms(previous_enabled, warn_only=previous_warn_only)
        torch.backends.cudnn.benchmark = previous_benchmark
        torch.backends.cudnn.deterministic = previous_cudnn_deterministic
        if previous_workspace is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = previous_workspace


def test_checkpoint_rejects_training_configuration_mismatch(tmp_path: Path) -> None:
    trainer = Trainer(model(), train_config(2), tmp_path / "source")
    stream = CyclingBatchStream(batches())
    trainer.fit(stream, until_step=1)
    checkpoint = trainer.save_checkpoint()
    mismatched = Trainer(
        model(),
        TrainConfig(
            max_steps=2, learning_rate=2e-3, weight_decay=0.0, device="cpu", precision="none"
        ),
        tmp_path / "target",
    )
    with pytest.raises(ValueError, match="training configuration"):
        mismatched.load_checkpoint(checkpoint, CyclingBatchStream(batches()))


def test_analysis_cadence_emits_layer_update_geometry(tmp_path: Path) -> None:
    config = TrainConfig(
        max_steps=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        device="cpu",
        precision="none",
        analysis_every_steps=1,
    )
    trainer = Trainer(model(), config, tmp_path)
    trainer.fit(CyclingBatchStream(batches()))
    records = (tmp_path / "telemetry.jsonl").read_text().splitlines()
    analysis = [record for record in records if "layer_analysis" in record]
    assert len(analysis) == 2
    assert "update_cosine" in analysis[-1]


def test_analysis_reference_survives_resume(tmp_path: Path) -> None:
    config = TrainConfig(
        max_steps=3,
        learning_rate=1e-3,
        weight_decay=0.0,
        device="cpu",
        precision="none",
        analysis_every_steps=1,
    )
    first = Trainer(model(), config, tmp_path / "first")
    stream = CyclingBatchStream(batches())
    first.fit(stream, until_step=1)
    checkpoint = first.save_checkpoint()
    resumed = Trainer(model(), config, tmp_path / "resumed")
    resumed_stream = CyclingBatchStream(batches())
    resumed.load_checkpoint(checkpoint, resumed_stream)
    resumed.fit(resumed_stream, until_step=2)
    analysis = [
        line
        for line in (tmp_path / "resumed" / "telemetry.jsonl").read_text().splitlines()
        if "layer_analysis" in line
    ]
    assert len(analysis) == 1
    metrics = json.loads(analysis[0])["per_layer"]
    assert any(values["update_norm"] > 0.0 for values in metrics.values())


def test_create_seeds_model_initialization_before_construction(tmp_path: Path) -> None:
    config = ModelConfig(
        architecture="gpt",
        vocab_size=32,
        max_seq_len=8,
        d_model=16,
        n_layers=2,
        n_heads=4,
        ffn_dim=32,
    )
    first = Trainer.create(config, train_config(2), tmp_path / "first")
    second = Trainer.create(config, train_config(2), tmp_path / "second")
    changed_seed = TrainConfig(
        max_steps=2, learning_rate=1e-3, weight_decay=0.0, device="cpu", precision="none", seed=124
    )
    third = Trainer.create(config, changed_seed, tmp_path / "third")
    for left, right in zip(first.model.parameters(), second.model.parameters(), strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert any(
        not torch.equal(left, changed)
        for left, changed in zip(first.model.parameters(), third.model.parameters(), strict=True)
    )


def test_rejects_mismatched_trajectory_metadata_hashes(tmp_path: Path) -> None:
    model_instance = model()
    config = train_config(2)
    invalid = SnapshotMetadata(
        run_id="run",
        checkpoint_id="initial",
        step=0,
        initialization_id="init",
        data_order_id="order",
        architecture_signature="gpt",
        corpus_signature="corpus",
        model_config_hash="incorrect",
        tokenizer_hash="tokenizer",
        code_signature="code",
        training_config_hash="incorrect",
    )
    with pytest.raises(ValueError, match="configuration hashes"):
        Trainer(model_instance, config, tmp_path, trajectory_metadata=invalid)


def verification_for(
    model_instance: DecoderOnlyTransformer,
    *,
    source_step: int,
    target_step: int,
    accepted: bool = True,
) -> CandidateVerification:
    candidate = {
        name: tensor.detach().clone() for name, tensor in model_instance.state_dict().items()
    }
    next(iter(candidate.values())).add_(0.001)
    forecast = ForecastResult(
        method="test",
        source_checkpoint_ids=("source",),
        source_steps=(source_step,),
        target_step=target_step,
        state_dict=candidate,
        diagnostics={},
    )
    prepared = prepare_forecast(model_instance, forecast)
    decision = CandidateDecision(
        accepted=accepted,
        reasons=() if accepted else ("rejected",),
        verification_loss_increase=0.0,
        post_leap_loss_increase=0.0,
        drift=FunctionDriftMetrics(0.0, 0.0, 0.0, 0.0, 1, 0),
        safety=ForecastSafetyReport(True, {}),
        candidate_state_hash=prepared.state_hash,
    )
    return CandidateVerification(
        prepared,
        decision,
        VerificationTiming(0.0, 0.0, 0.0),
        baseline_comparator="current_checkpoint",
    )


def transport_config(max_steps: int = 4) -> TrainConfig:
    return TrainConfig(
        max_steps=max_steps,
        learning_rate=1e-3,
        weight_decay=0.0,
        device="cpu",
        precision="none",
        seed=123,
        gradient_accumulation_steps=2,
    )


def test_verified_transport_advances_virtual_data_without_misreporting_processed_tokens(
    tmp_path: Path,
) -> None:
    trainer = Trainer(model(), transport_config(), tmp_path)
    stream = CyclingBatchStream(batches())
    trainer.fit(stream, until_step=1)
    processed_before = trainer.state.data_tokens_seen
    verification = verification_for(trainer.model, source_step=1, target_step=3)
    skipped = trainer.apply_verified_transport(verification, stream)
    assert skipped.batch_count == 4
    assert skipped.data_token_count == 58
    assert trainer.state.global_step == 3
    assert trainer.state.optimizer_steps_executed == 1
    assert trainer.state.virtual_steps_skipped == 2
    assert trainer.state.micro_batches_skipped == 4
    assert trainer.state.data_tokens_skipped == 58
    assert trainer.state.data_tokens_seen == processed_before
    assert trainer.scheduler.last_epoch == 3


def test_verified_transport_retain_and_reset_optimizer_state(tmp_path: Path) -> None:
    retained = Trainer(model(), transport_config(), tmp_path / "retained")
    retained_stream = CyclingBatchStream(batches())
    retained.fit(retained_stream, until_step=1)
    assert retained.optimizer.state
    retained.apply_verified_transport(
        verification_for(retained.model, source_step=1, target_step=2), retained_stream
    )
    assert retained.optimizer.state

    reset = Trainer(model(), transport_config(), tmp_path / "reset")
    reset_stream = CyclingBatchStream(batches())
    reset.fit(reset_stream, until_step=1)
    reset.apply_verified_transport(
        verification_for(reset.model, source_step=1, target_step=2),
        reset_stream,
        optimizer_state_policy="reset",
    )
    assert not reset.optimizer.state


def test_rejected_or_mismatched_transport_leaves_state_and_stream_unchanged(tmp_path: Path) -> None:
    trainer = Trainer(model(), transport_config(), tmp_path)
    stream = CyclingBatchStream(batches())
    trainer.fit(stream, until_step=1)
    before = {name: tensor.detach().clone() for name, tensor in trainer.model.state_dict().items()}
    cursor = stream.state_dict()
    rejected = verification_for(trainer.model, source_step=1, target_step=2, accepted=False)
    with pytest.raises(ValueError, match="rejected"):
        trainer.apply_verified_transport(rejected, stream)
    assert stream.state_dict() == cursor
    for name, tensor in trainer.model.state_dict().items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)

    mismatched = verification_for(trainer.model, source_step=0, target_step=2)
    with pytest.raises(ValueError, match="anchored"):
        trainer.apply_verified_transport(mismatched, stream)
    assert stream.state_dict() == cursor

    candidate_mismatch = verification_for(trainer.model, source_step=1, target_step=2)
    invalid_hash = replace(
        candidate_mismatch,
        decision=replace(candidate_mismatch.decision, candidate_state_hash="different"),
    )
    with pytest.raises(ValueError, match="hash"):
        trainer.apply_verified_transport(invalid_hash, stream)
    assert stream.state_dict() == cursor


def test_post_transport_checkpoint_resume_is_exact(tmp_path: Path) -> None:
    torch.manual_seed(77)
    initial = model()
    initial_state = {name: tensor.detach().clone() for name, tensor in initial.state_dict().items()}

    direct_model = model()
    direct_model.load_state_dict(initial_state)
    direct = Trainer(direct_model, transport_config(), tmp_path / "direct")
    direct_stream = CyclingBatchStream(batches())
    direct.fit(direct_stream, until_step=1)
    direct.apply_verified_transport(
        verification_for(direct.model, source_step=1, target_step=3), direct_stream
    )
    direct.fit(direct_stream)

    split_model = model()
    split_model.load_state_dict(initial_state)
    split = Trainer(split_model, transport_config(), tmp_path / "split")
    split_stream = CyclingBatchStream(batches())
    split.fit(split_stream, until_step=1)
    split.apply_verified_transport(
        verification_for(split.model, source_step=1, target_step=3), split_stream
    )
    checkpoint = split.save_checkpoint()
    resumed = Trainer(model(), transport_config(), tmp_path / "resumed")
    resumed_stream = CyclingBatchStream(batches())
    resumed.load_checkpoint(checkpoint, resumed_stream)
    resumed.fit(resumed_stream)
    assert resumed.state.global_step == direct.state.global_step
    assert resumed.state.optimizer_steps_executed == direct.state.optimizer_steps_executed
    assert resumed.state.virtual_steps_skipped == direct.state.virtual_steps_skipped
    assert resumed.state.micro_batches_skipped == direct.state.micro_batches_skipped
    assert resumed.state.data_tokens_seen == direct.state.data_tokens_seen
    assert resumed.state.data_tokens_skipped == direct.state.data_tokens_skipped
    assert resumed.state.supervised_tokens_seen == direct.state.supervised_tokens_seen
    for expected, actual in zip(direct.model.parameters(), resumed.model.parameters(), strict=True):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)


def test_gradient_accumulation_is_token_weighted(tmp_path: Path) -> None:
    torch.manual_seed(91)
    initial = model()
    initial_state = {name: tensor.detach().clone() for name, tensor in initial.state_dict().items()}
    input_a = torch.randint(0, 32, (1, 8))
    input_b = torch.randint(0, 32, (1, 8))
    target_a = torch.full((1, 8), -100, dtype=torch.long)
    target_a[:, 0] = 1
    target_b = torch.randint(0, 32, (1, 8))
    microbatches = (
        {"input_ids": input_a, "targets": target_a},
        {"input_ids": input_b, "targets": target_b},
    )
    accumulated_model = model()
    accumulated_model.load_state_dict(initial_state)
    accumulated = Trainer(
        accumulated_model,
        TrainConfig(
            max_steps=1,
            learning_rate=1e-3,
            weight_decay=0.0,
            device="cpu",
            precision="none",
            gradient_accumulation_steps=2,
        ),
        tmp_path / "accumulated",
    )
    accumulated.fit(CyclingBatchStream(microbatches))

    combined_model = model()
    combined_model.load_state_dict(initial_state)
    combined = Trainer(
        combined_model,
        TrainConfig(
            max_steps=1, learning_rate=1e-3, weight_decay=0.0, device="cpu", precision="none"
        ),
        tmp_path / "combined",
    )
    combined.fit(
        CyclingBatchStream(
            (
                {
                    "input_ids": torch.cat((input_a, input_b)),
                    "targets": torch.cat((target_a, target_b)),
                },
            )
        )
    )
    for expected, actual in zip(
        combined.model.parameters(), accumulated.model.parameters(), strict=True
    ):
        torch.testing.assert_close(expected, actual, rtol=1e-4, atol=1e-6)


def test_per_example_loss_hook_requires_one_identifier_per_row(tmp_path: Path) -> None:
    invalid_batch = dict(batches()[0])
    invalid_batch["example_ids"] = ["only-one"]
    trainer = Trainer(model(), train_config(1), tmp_path, per_example_loss_hook=lambda *_: None)
    with pytest.raises(ValueError, match="batch row count"):
        trainer.fit(CyclingBatchStream((invalid_batch,)))


def test_checkpoint_rejects_untrusted_and_invalid_training_state_without_mutation(
    tmp_path: Path,
) -> None:
    source = Trainer(model(), train_config(2), tmp_path / "source")
    source_stream = CyclingBatchStream(batches())
    source.fit(source_stream, until_step=1)
    checkpoint = source.save_checkpoint()
    target = Trainer(model(), train_config(2), tmp_path / "target")
    target_stream = CyclingBatchStream(batches())
    target_before = {
        name: tensor.detach().clone() for name, tensor in target.model.state_dict().items()
    }

    untrusted_path = tmp_path / "untrusted.pt"
    torch.save(_UntrustedPayload(), untrusted_path)
    with pytest.raises(ValueError, match="restricted"):
        target.load_checkpoint(untrusted_path, target_stream)

    malformed = torch.load(checkpoint, weights_only=True)
    malformed["training_state"]["global_step"] = -1
    malformed_path = tmp_path / "malformed.pt"
    torch.save(malformed, malformed_path)
    with pytest.raises(ValueError, match="training_state"):
        target.load_checkpoint(malformed_path, target_stream)
    assert target_stream.state_dict() == {"index": 0}
    for name, tensor in target.model.state_dict().items():
        torch.testing.assert_close(tensor, target_before[name], rtol=0, atol=0)


def _assert_nested_equal(expected: Any, actual: Any) -> None:
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert expected.keys() == actual.keys()
        for key in expected:
            _assert_nested_equal(expected[key], actual[key])
    elif isinstance(expected, (list, tuple)):
        assert isinstance(actual, type(expected))
        assert len(expected) == len(actual)
        for left, right in zip(expected, actual, strict=True):
            _assert_nested_equal(left, right)
    else:
        assert expected == actual


def _assert_transport_rollback(
    trainer: Trainer,
    stream: CyclingBatchStream,
    before_model: dict[str, torch.Tensor],
    before_state: dict[str, object],
    before_optimizer: dict[str, object],
    before_scheduler: dict[str, object],
    before_scaler: dict[str, object],
    before_cursor: dict[str, int],
) -> None:
    assert asdict(trainer.state) == before_state
    assert stream.state_dict() == before_cursor
    _assert_nested_equal(before_optimizer, trainer.optimizer.state_dict())
    _assert_nested_equal(before_scheduler, trainer.scheduler.state_dict())
    _assert_nested_equal(before_scaler, trainer.scaler.state_dict())
    for name, tensor in trainer.model.state_dict().items():
        torch.testing.assert_close(tensor, before_model[name], rtol=0, atol=0)


def _transport_backups(
    trainer: Trainer, stream: CyclingBatchStream
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, int],
]:
    return (
        {name: tensor.detach().clone() for name, tensor in trainer.model.state_dict().items()},
        asdict(trainer.state),
        trainer.optimizer.state_dict(),
        trainer.scheduler.state_dict(),
        trainer.scaler.state_dict(),
        stream.state_dict(),
    )


def test_verified_transport_rolls_back_all_state_on_stream_or_scheduler_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FailingStream(CyclingBatchStream):
        def skip_batches(self, count: int) -> object:
            super().skip_batches(1)
            raise RuntimeError("injected stream failure")

    stream_trainer = Trainer(model(), transport_config(), tmp_path / "stream")
    stream = FailingStream(batches())
    stream_trainer.fit(stream, until_step=1)
    before = _transport_backups(stream_trainer, stream)
    with pytest.raises(RuntimeError, match="stream failure"):
        stream_trainer.apply_verified_transport(
            verification_for(stream_trainer.model, source_step=1, target_step=2), stream
        )
    _assert_transport_rollback(stream_trainer, stream, *before)

    scheduler_trainer = Trainer(model(), transport_config(), tmp_path / "scheduler")
    scheduler_stream = CyclingBatchStream(batches())
    scheduler_trainer.fit(scheduler_stream, until_step=1)

    def fail_scheduler() -> None:
        raise RuntimeError("injected scheduler failure")

    monkeypatch.setattr(scheduler_trainer.scheduler, "step", fail_scheduler)
    before = _transport_backups(scheduler_trainer, scheduler_stream)
    with pytest.raises(RuntimeError, match="scheduler failure"):
        scheduler_trainer.apply_verified_transport(
            verification_for(scheduler_trainer.model, source_step=1, target_step=2),
            scheduler_stream,
        )
    _assert_transport_rollback(scheduler_trainer, scheduler_stream, *before)


def test_declared_capture_steps_save_only_selected_milestones(tmp_path: Path) -> None:
    config = TrainConfig(
        max_steps=3,
        learning_rate=1e-3,
        weight_decay=0.0,
        device="cpu",
        precision="none",
        checkpoint_steps=(2,),
        trajectory_capture_steps=(3,),
    )
    trainer = Trainer(
        model(),
        config,
        tmp_path,
        trajectory_metadata=trajectory_metadata(model().config, config),
    )
    trainer.fit(CyclingBatchStream(batches()))
    assert (tmp_path / "checkpoints" / "step_00000002.pt").is_file()
    assert (tmp_path / "trajectory" / "step_00000003.pt").is_file()
    assert not (tmp_path / "checkpoints" / "step_00000001.pt").exists()


def test_capture_steps_must_be_sorted_unique_and_in_range() -> None:
    common = {"max_steps": 3, "learning_rate": 1e-3}
    with pytest.raises(ValueError, match="unique and sorted"):
        TrainConfig(**common, checkpoint_steps=(2, 1))
    with pytest.raises(ValueError, match=r"\[1, max_steps\]"):
        TrainConfig(**common, trajectory_capture_steps=(4,))
