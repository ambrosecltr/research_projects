from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from poetry50m.evaluation.schema import CostRecord
from poetry50m.model import DecoderOnlyTransformer, ModelConfig
from poetry50m.trajectory import state_dict_hash
from poetry50m.trajectory.application import apply_accepted_candidate
from poetry50m.trajectory.forecast import (
    ForecastResult,
    LinearForecastConfig,
    linear_finite_difference,
)
from poetry50m.trajectory.gates import (
    AnchorOutputs,
    CandidateAcceptanceGates,
    CandidateDecision,
    ForecastSafetyConfig,
    decide_candidate,
    fixed_anchor_function_drift,
    forecast_safety_report,
    future_target_distillation_loss,
)
from poetry50m.trajectory.ledgers import (
    AnalysisLedger,
    CostLedger,
    CostTotals,
    CpuCost,
    DecisionLedger,
    write_json_ledger,
)
from poetry50m.trajectory.preparation import declared_normalization_axes, prepare_forecast
from poetry50m.trajectory.verification import (
    CandidateVerification,
    VerificationBatch,
    VerificationTiming,
    decoder_loss_evaluator,
    verify_candidate,
)

from .conftest import make_snapshot


def _anchors(logits: torch.Tensor) -> AnchorOutputs:
    return AnchorOutputs(logits={"logits": logits}, representations={})


def _accepted_decision() -> CandidateDecision:
    current = {"weight": torch.tensor([1.0, 2.0])}
    candidate = {"weight": torch.tensor([1.1, 2.1])}
    safety = forecast_safety_report(
        current, candidate, ForecastSafetyConfig(max_relative_tensor_norm=2.0)
    )
    drift = fixed_anchor_function_drift(
        _anchors(torch.zeros(1, 1, 2)), _anchors(torch.zeros(1, 1, 2))
    )
    return decide_candidate(
        baseline_verification_loss=1.0,
        candidate_verification_loss=1.0,
        baseline_post_leap_loss=1.0,
        candidate_post_leap_loss=1.0,
        drift=drift,
        safety=safety,
        gates=CandidateAcceptanceGates(0.01, 0.01, 0.001, 0.01, 0.001),
    )


def test_anchor_semantics_function_loss_and_finite_metrics() -> None:
    logits = torch.tensor([[[2.0, 0.0]]])
    representations = torch.tensor([[1.0, -1.0]])
    drift = fixed_anchor_function_drift(
        AnchorOutputs(logits={"logits": logits}, representations={"residual": representations}),
        AnchorOutputs(logits={"logits": logits}, representations={"residual": representations}),
    )
    assert drift.mean_squared_error == 0.0
    assert drift.cosine_distance == 0.0
    assert drift.distribution_count == 1
    with pytest.raises(ValueError, match="cannot be empty"):
        AnchorOutputs(logits={}, representations={})
    with pytest.raises(ValueError, match="finite"):
        CandidateAcceptanceGates(float("nan"), 0.1, 0.1, 0.1, 0.1)
    with pytest.raises(TypeError, match="finite"):
        CandidateAcceptanceGates(True, 0.1, 0.1, 0.1, 0.1)

    current_logits = torch.tensor([[[0.5, -0.5]]], requires_grad=True)
    loss = future_target_distillation_loss(current_logits, logits, mse_weight=0.1)
    loss.backward()
    assert current_logits.grad is not None
    repeated_loss = future_target_distillation_loss(
        current_logits.detach().repeat(1, 2, 1), logits.repeat(1, 2, 1), mse_weight=0.1
    )
    torch.testing.assert_close(loss.detach(), repeated_loss)


def test_norm_collapse_and_near_zero_safety_handling() -> None:
    collapse = forecast_safety_report(
        {"weight": torch.ones(4)},
        {"weight": torch.full((4,), 0.01)},
        ForecastSafetyConfig(min_relative_tensor_norm=0.5),
    )
    assert not collapse.accepted
    near_zero = forecast_safety_report(
        {"weight": torch.zeros(4)},
        {"weight": torch.full((4,), 1e-12)},
        ForecastSafetyConfig(),
    )
    assert near_zero.accepted
    assert near_zero.tensors["weight"].near_zero_reference
    with pytest.raises(ValueError, match="finite"):
        ForecastSafetyConfig(max_relative_tensor_norm=float("nan"))
    with pytest.raises(TypeError, match="finite"):
        ForecastSafetyConfig(max_relative_tensor_norm=True)


def test_cost_ledger_all_units_and_atomic_decision_json(tmp_path: Path) -> None:
    def record(name: str, seconds: float) -> CostRecord:
        return CostRecord(
            "r0",
            name,
            0,
            0,
            seconds * 2,
            seconds,
            seconds / 10.0,
            seconds * 2,
        )

    ledger = CostLedger(
        reference=record("reference", 100.0),
        analysis=record("analysis", 20.0),
        checkpoint_io=record("io", 10.0),
        verification_per_replay=record("verify", 5.0),
        replay=record("replay", 50.0),
        baseline_replay=record("baseline", 100.0),
        actual_peak_working_memory_bytes=4_096,
        current_working_memory_bytes=2_048,
        peak_memory_semantics="test_peak",
        checkpoint_io_wall_seconds=2.5,
        snapshot_bytes_read=2_048,
        snapshot_bytes_written=1_024,
        reference_cpu=CpuCost(20.0),
        analysis_cpu=CpuCost(12.0),
        checkpoint_io_cpu=CpuCost(2.0),
        verification_cpu_per_replay=CpuCost(1.0),
        replay_cpu=CpuCost(4.0),
        baseline_replay_cpu=CpuCost(10.0),
    )
    assert ledger.total_discovery.accelerator_seconds == 130.0
    assert ledger.accelerated_per_replay.wall_seconds == 110.0
    expected_cpu_seconds = (20.0 + 12.0 + 2.0 + 2 * (4.0 + 1.0)) / 2
    assert ledger.amortized(2).cpu_seconds == expected_cpu_seconds
    assert ledger.break_even_uses()["accelerator_seconds"] == 3
    assert ledger.break_even_uses()["device_active_wall_seconds"] == 3
    assert ledger.break_even_uses()["cpu_seconds"] == 7
    assert ledger.break_even_uses()["estimated_cost_usd"] == 3
    assert ledger._break_even(100.0, 100.0, 50.0) == 3
    with pytest.raises(ValueError, match="finite"):
        CpuCost(float("inf"))
    with pytest.raises(ValueError, match="finite"):
        CostTotals(float("nan"), 1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="finite"):
        CostTotals(None, True, 1.0, None)
    with pytest.raises(ValueError, match="finite"):
        CostTotals(None, 1.0, 1.0, None, True)
    with pytest.raises(ValueError, match="finite"):
        replace(ledger, replay=record("bad", float("nan")))
    unknown_accelerator_price = CostRecord("r0", "unknown", 0, 0, 1.0, 1.0, None)
    assert (
        replace(ledger, analysis=unknown_accelerator_price).total_discovery.estimated_cost_usd
        is None
    )
    unknown_accelerator = replace(
        ledger,
        analysis=replace(ledger.analysis, accelerator_seconds=None),
        actual_peak_working_memory_bytes=None,
        current_working_memory_bytes=1_024,
        peak_memory_semantics="mps_peak_unavailable_current_reported",
    )
    assert unknown_accelerator.total_discovery.accelerator_seconds is None
    assert unknown_accelerator.amortized(2).accelerator_seconds is None
    assert unknown_accelerator.break_even_uses()["accelerator_seconds"] is None
    assert unknown_accelerator.break_even_uses()["wall_seconds"] == 3
    assert unknown_accelerator.break_even_uses()["cpu_seconds"] == 7
    assert unknown_accelerator.to_mapping()["actual_peak_working_memory_bytes"] is None
    assert unknown_accelerator.to_mapping()["current_working_memory_bytes"] == 1_024
    with pytest.raises(ValueError, match="resource"):
        replace(ledger, actual_peak_working_memory_bytes=True)
    with pytest.raises(ValueError, match="resource"):
        replace(ledger, current_working_memory_bytes=True)
    with pytest.raises(ValueError, match="resource"):
        replace(ledger, checkpoint_io_wall_seconds=True)
    with pytest.raises(ValueError, match="non-empty string"):
        replace(ledger, peak_memory_semantics="")
    path = tmp_path / "decision.json"
    write_json_ledger(path, DecisionLedger("r0", "s1", _accepted_decision()))
    assert json.loads(path.read_text())["decision"]["accepted"]


def test_analysis_ledger_distinguishes_unknown_memory_from_zero() -> None:
    forecast = ForecastResult("synthetic", ("s0",), (0,), 1, {}, {})
    unknown = AnalysisLedger("r0", forecast, 0.0, 0.0, 0.0, None, 0, 0)
    zero = replace(unknown, actual_peak_working_memory_bytes=0)
    assert unknown.to_mapping()["actual_peak_working_memory_bytes"] is None
    assert zero.to_mapping()["actual_peak_working_memory_bytes"] == 0
    with pytest.raises(ValueError, match="resource"):
        replace(unknown, snapshot_bytes_read=True)


def test_ngpt_preparation_retracts_every_declared_axis_before_application() -> None:
    model = DecoderOnlyTransformer(
        ModelConfig(
            "ngpt", vocab_size=16, max_seq_len=8, d_model=8, n_layers=2, n_heads=2, ffn_dim=16
        )
    )
    raw_state = {
        name: tensor.detach().clone() * 7.0
        if tensor.is_floating_point()
        else tensor.detach().clone()
        for name, tensor in model.state_dict().items()
    }
    forecast = ForecastResult("synthetic", ("s0",), (0,), 1, raw_state, {})
    prepared = prepare_forecast(model, forecast)
    axes = declared_normalization_axes(model)
    expected_axes = {
        "token_embedding.weight": -1,
        "blocks.0.attention.query.weight": 1,
        "blocks.0.attention.key.weight": 1,
        "blocks.0.attention.value.weight": 1,
        "blocks.0.attention.output.weight": 0,
        "blocks.0.mlp.in_projection.weight": 1,
        "blocks.0.mlp.out_projection.weight": 0,
        "blocks.1.attention.query.weight": 1,
        "blocks.1.attention.key.weight": 1,
        "blocks.1.attention.value.weight": 1,
        "blocks.1.attention.output.weight": 0,
        "blocks.1.mlp.in_projection.weight": 1,
        "blocks.1.mlp.out_projection.weight": 0,
    }
    assert axes == expected_axes
    assert prepared.retracted_axes == expected_axes
    for name, axis in expected_axes.items():
        norms = torch.linalg.vector_norm(prepared.state_dict[name], dim=axis)
        torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5)


def test_tiny_model_verification_prepares_then_decides_then_applies() -> None:
    torch.manual_seed(0)
    model = DecoderOnlyTransformer(
        ModelConfig(
            "gpt", vocab_size=16, max_seq_len=8, d_model=8, n_layers=1, n_heads=2, ffn_dim=16
        )
    )
    state0 = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    state1 = {
        name: tensor.detach().clone() + 0.001
        if tensor.is_floating_point()
        else tensor.detach().clone()
        for name, tensor in state0.items()
    }
    forecast = linear_finite_difference(
        make_snapshot(checkpoint_id="s0", step=0, state_dict=state0),
        make_snapshot(checkpoint_id="s2", step=2, state_dict=state1),
        target_step=4,
        config=LinearForecastConfig(),
    )
    batch = VerificationBatch(torch.tensor([[1, 2, 3]]), torch.tensor([[2, 3, 4]]))

    def anchor_evaluator(current: torch.nn.Module) -> AnchorOutputs:
        assert isinstance(current, DecoderOnlyTransformer)
        return _anchors(current(batch.input_ids).logits)

    verification = verify_candidate(
        module=model,
        forecast=forecast,
        batches=(batch,),
        anchor_evaluator=anchor_evaluator,
        gates=CandidateAcceptanceGates(10.0, 10.0, 10.0, 10.0, 10.0),
        safety_config=ForecastSafetyConfig(
            max_relative_tensor_norm=10.0, min_relative_tensor_norm=0.0
        ),
    )
    assert verification.decision.accepted
    assert verification.baseline_comparator == "current_checkpoint"
    apply_accepted_candidate(model, verification)
    assert state_dict_hash(verification.prepared.state_dict) == verification.prepared.state_hash
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, verification.prepared.state_dict[name])

    rejected = CandidateVerification(
        verification.prepared,
        decide_candidate(
            baseline_verification_loss=1.0,
            candidate_verification_loss=2.0,
            baseline_post_leap_loss=1.0,
            candidate_post_leap_loss=2.0,
            drift=fixed_anchor_function_drift(
                _anchors(torch.zeros(1, 1, 2)), _anchors(torch.zeros(1, 1, 2))
            ),
            safety=forecast_safety_report(
                state1,
                verification.prepared.state_dict,
                ForecastSafetyConfig(max_relative_tensor_norm=10.0),
            ),
            gates=CandidateAcceptanceGates(0.0, 0.0, 1.0, 1.0, 1.0),
        ),
        VerificationTiming(0.0, 0.0, 0.0),
        "current_checkpoint",
    )
    with pytest.raises(ValueError, match="refusing"):
        apply_accepted_candidate(model, rejected)


def test_mutating_post_leap_callback_is_isolated_and_uses_both_actual_losses() -> None:
    model = DecoderOnlyTransformer(
        ModelConfig(
            "gpt", vocab_size=16, max_seq_len=8, d_model=8, n_layers=1, n_heads=2, ffn_dim=16
        )
    )
    model.train()
    original = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    forecast = ForecastResult("identity", ("s0",), (0,), 1, original, {})
    batch = VerificationBatch(torch.tensor([[1, 2, 3]]), torch.tensor([[2, 3, 4]]))
    calls: list[bool] = []

    def anchors(current: torch.nn.Module) -> AnchorOutputs:
        assert isinstance(current, DecoderOnlyTransformer)
        return _anchors(current(batch.input_ids).logits)

    def mutating_post_leap(current: torch.nn.Module) -> float:
        calls.append(current.training)
        with torch.no_grad():
            next(current.parameters()).add_(1.0)
        return 1.0 if len(calls) == 1 else 3.0

    verification = verify_candidate(
        module=model,
        forecast=forecast,
        batches=(batch,),
        anchor_evaluator=anchors,
        gates=CandidateAcceptanceGates(10.0, 1.0, 10.0, 10.0, 10.0),
        safety_config=ForecastSafetyConfig(max_relative_tensor_norm=10.0),
        post_leap_evaluator=mutating_post_leap,
    )
    assert verification.baseline_comparator == "current_checkpoint"
    assert verification.decision.post_leap_loss_increase == 2.0
    assert not verification.decision.accepted
    assert "post-leap loss increase exceeded gate" in verification.decision.reasons
    assert calls == [False, False]
    assert model.training
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, original[name])


def test_verification_records_explicit_continued_baseline_comparator() -> None:
    model = DecoderOnlyTransformer(
        ModelConfig(
            "gpt", vocab_size=16, max_seq_len=8, d_model=8, n_layers=1, n_heads=2, ffn_dim=16
        )
    )
    state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    forecast = ForecastResult("identity", ("s0",), (0,), 1, state, {})
    batch = VerificationBatch(torch.tensor([[1, 2, 3]]), torch.tensor([[2, 3, 4]]))

    def anchors(current: torch.nn.Module) -> AnchorOutputs:
        assert isinstance(current, DecoderOnlyTransformer)
        return _anchors(current(batch.input_ids).logits)

    verification = verify_candidate(
        module=model,
        forecast=forecast,
        batches=(batch,),
        anchor_evaluator=anchors,
        gates=CandidateAcceptanceGates(1.0, 1.0, 1.0, 1.0, 1.0),
        safety_config=ForecastSafetyConfig(),
        continued_baseline_state=state,
    )
    assert verification.baseline_comparator == "continued_baseline"


def test_decoder_loss_evaluator_is_deterministic_and_restores_training_mode() -> None:
    model = DecoderOnlyTransformer(
        ModelConfig(
            "gpt",
            vocab_size=16,
            max_seq_len=8,
            d_model=8,
            n_layers=1,
            n_heads=2,
            ffn_dim=16,
            dropout=0.5,
        )
    )
    model.train()
    batches = (VerificationBatch(torch.tensor([[1, 2, 3]]), torch.tensor([[2, 3, 4]])),)
    first_loss = decoder_loss_evaluator(model, batches)
    second_loss = decoder_loss_evaluator(model, batches)
    assert model.training
    assert first_loss == second_loss


@pytest.mark.parametrize("fault", ("names", "shape", "dtype", "nonfinite"))
def test_invalid_continued_baseline_is_rejected_before_verification_mutates_model(
    fault: str,
) -> None:
    model = DecoderOnlyTransformer(
        ModelConfig(
            "gpt", vocab_size=16, max_seq_len=8, d_model=8, n_layers=1, n_heads=2, ffn_dim=16
        )
    )
    model.train()
    original = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    forecast = ForecastResult("identity", ("s0",), (0,), 1, original, {})
    continued = {name: tensor.detach().clone() for name, tensor in original.items()}
    first_name = next(iter(continued))
    if fault == "names":
        continued = dict(tuple(continued.items())[1:])
        expected = "names or order"
    elif fault == "shape":
        continued[first_name] = torch.ones(1, dtype=continued[first_name].dtype)
        expected = "shape"
    elif fault == "dtype":
        continued[first_name] = continued[first_name].to(torch.float64)
        expected = "dtype"
    else:
        continued[first_name].fill_(float("inf"))
        expected = "finite"

    with pytest.raises(ValueError, match=expected):
        verify_candidate(
            module=model,
            forecast=forecast,
            batches=(VerificationBatch(torch.tensor([[1, 2, 3]]), torch.tensor([[2, 3, 4]])),),
            anchor_evaluator=lambda current: _anchors(current(torch.tensor([[1, 2, 3]])).logits),
            gates=CandidateAcceptanceGates(1.0, 1.0, 1.0, 1.0, 1.0),
            safety_config=ForecastSafetyConfig(),
            continued_baseline_state=continued,
        )
    assert model.training
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, original[name])
