from __future__ import annotations

from pathlib import Path

import pytest
import torch

from genome.architecture import graph_from_state
from genome.mgp import (
    Component,
    FitConfig,
    ModelGenomeProgram,
    TensorProgram,
    audit_program,
    execute_program,
    fit_low_rank_program,
    load_program,
    save_program,
)
from genome.mgp.policy import ProgramPolicy
from genome.mgp.serialize import serialized_program_bytes
from genome.state import direct_fp16_delta_bytes


def states() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    torch.manual_seed(7)
    w0 = {
        "layers.0.weight": torch.randn(32, 24),
        "layers.0.bias": torch.randn(32),
        "final_norm.weight": torch.ones(24),
    }
    wt = {
        "layers.0.weight": w0["layers.0.weight"] + torch.randn(32, 3) @ torch.randn(3, 24) * 0.1,
        "layers.0.bias": w0["layers.0.bias"] + torch.randn(32) * 0.01,
        "final_norm.weight": w0["final_norm.weight"] + torch.randn(24) * 0.01,
    }
    return w0, wt


def test_only_compact_primitives_exist() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        Component("DENSE_DELTA")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unsupported"):
        Component("NEURAL_BLOCK_FIELD")  # type: ignore[arg-type]


def test_fit_serialize_execute_and_audit(tmp_path: Path) -> None:
    w0, wt = states()
    graph = graph_from_state(w0, family="toy", config={})
    program, payloads = fit_low_rank_program(
        w0, wt, graph, config=FitConfig(budget_fraction=0.5, max_rank=4)
    )
    accounting = save_program(tmp_path / "program", program, payloads)
    loaded, loaded_payloads, manifest = load_program(tmp_path / "program")
    candidate = execute_program(w0, loaded, loaded_payloads)
    assert set(candidate) == set(w0)
    assert all(torch.isfinite(value).all() for value in candidate.values())
    audit = audit_program(
        loaded,
        loaded_payloads,
        direct_fp16_delta_bytes=max(direct_fp16_delta_bytes(wt), accounting["total_bytes"] * 20),
        artifact_directory=tmp_path / "program",
        policy=ProgramPolicy(primary_fraction=0.10, exploratory_fraction=0.20),
    )
    assert audit.accepted_structure
    assert audit.serialized
    assert manifest["program_id"] == accounting["program_id"]


def test_fit_can_account_for_complete_serialized_size(tmp_path: Path) -> None:
    torch.manual_seed(11)
    w0 = {"weight": torch.randn(256, 256)}
    wt = {"weight": w0["weight"] + torch.randn(256, 8) @ torch.randn(8, 256)}
    graph = graph_from_state(w0, family="toy", config={})
    budget_fraction = 0.10
    program, payloads = fit_low_rank_program(
        w0,
        wt,
        graph,
        config=FitConfig(
            budget_fraction=budget_fraction,
            max_rank=8,
            account_for_serialization=True,
        ),
    )

    direct_bytes = direct_fp16_delta_bytes(wt)
    estimated_bytes = serialized_program_bytes(program, payloads)
    accounting = save_program(tmp_path / "program", program, payloads)

    assert estimated_bytes == accounting["total_bytes"]
    assert accounting["total_bytes"] <= direct_bytes * budget_fraction


def test_policy_rejects_noncompact_low_rank() -> None:
    program = ModelGenomeProgram(
        architecture_id="a",
        base_state_id="b",
        tensors=(
            TensorProgram(
                name="matrix",
                shape=(4, 4),
                components=(
                    Component("BASE_COPY"),
                    Component(
                        "LOW_RANK", payload={"left": "l", "right": "r"}, arguments={"rank": 4}
                    ),
                ),
            ),
        ),
    )
    audit = audit_program(
        program,
        {"l": torch.zeros(4, 4), "r": torch.zeros(4, 4)},
        direct_fp16_delta_bytes=32,
    )
    assert not audit.accepted_structure
    assert any("non_compact_low_rank" in reason for reason in audit.reasons)
