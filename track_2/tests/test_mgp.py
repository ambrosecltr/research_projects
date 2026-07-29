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
from genome.mgp.fit import TrainableProgram, _relative_anchor_loss, _token_mean_kl
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


def test_fit_can_reserve_a_minimum_rank_for_every_matrix() -> None:
    torch.manual_seed(12)
    w0 = {
        "a": torch.randn(32, 24),
        "b": torch.randn(24, 32),
    }
    wt = {name: value + torch.randn_like(value) for name, value in w0.items()}
    graph = graph_from_state(w0, family="toy", config={})

    program, _ = fit_low_rank_program(
        w0,
        wt,
        graph,
        config=FitConfig(
            budget_fraction=0.20,
            max_rank=4,
            minimum_matrix_rank=1,
        ),
    )

    for tensor in program.tensors:
        component = next(item for item in tensor.components if item.primitive == "LOW_RANK")
        assert component.arguments["rank"] >= 1


def test_rank_balanced_fit_keeps_matrix_ranks_close() -> None:
    torch.manual_seed(13)
    w0 = {
        "a": torch.randn(32, 24),
        "b": torch.randn(32, 24),
    }
    wt = {name: value + torch.randn_like(value) for name, value in w0.items()}
    graph = graph_from_state(w0, family="toy", config={})

    program, _ = fit_low_rank_program(
        w0,
        wt,
        graph,
        config=FitConfig(
            budget_fraction=0.20,
            max_rank=4,
            allocation_strategy="rank_balanced",
        ),
    )

    ranks = []
    for tensor in program.tensors:
        component = next(item for item in tensor.components if item.primitive == "LOW_RANK")
        ranks.append(int(component.arguments["rank"]))
    assert min(ranks) > 0
    assert max(ranks) - min(ranks) <= 1


def test_hadamard_scale_runtime_uses_one_value_per_row_and_column() -> None:
    base = {"matrix": torch.arange(1, 10, dtype=torch.float32).reshape(3, 3)}
    program = ModelGenomeProgram(
        architecture_id="a",
        base_state_id="b",
        tensors=(
            TensorProgram(
                name="matrix",
                shape=(3, 3),
                components=(
                    Component("BASE_COPY"),
                    Component(
                        "HADAMARD_SCALE",
                        payload={"row": "matrix.row", "column": "matrix.column"},
                    ),
                ),
            ),
        ),
    )
    payloads = {
        "matrix.row": torch.tensor([0.1, 0.2, 0.3]),
        "matrix.column": torch.tensor([-0.1, 0.0, 0.1]),
    }

    candidate = execute_program(base, program, payloads)
    expected_scale = payloads["matrix.row"].unsqueeze(1) + payloads["matrix.column"].unsqueeze(0)

    assert torch.allclose(
        candidate["matrix"],
        base["matrix"] + base["matrix"] * expected_scale,
    )
    audit = audit_program(
        program,
        payloads,
        direct_fp16_delta_bytes=18,
    )
    assert audit.accepted_structure


def test_fit_hadamard_scale_removes_base_relative_matrix_change() -> None:
    torch.manual_seed(14)
    base_matrix = torch.randn(32, 24)
    row = torch.linspace(-0.1, 0.1, 32)
    column = torch.linspace(0.05, -0.05, 24)
    w0 = {"matrix": base_matrix}
    wt = {"matrix": base_matrix + base_matrix * (row.unsqueeze(1) + column.unsqueeze(0))}
    graph = graph_from_state(w0, family="toy", config={})

    program, payloads = fit_low_rank_program(
        w0,
        wt,
        graph,
        config=FitConfig(
            budget_fraction=0.50,
            max_rank=4,
            matrix_scaling=True,
            svd_method="exact",
        ),
    )
    candidate = execute_program(w0, program, payloads)
    relative_error = (candidate["matrix"] - wt["matrix"]).square().sum() / (
        wt["matrix"] - w0["matrix"]
    ).square().sum()

    assert any(
        component.primitive == "HADAMARD_SCALE" for component in program.tensors[0].components
    )
    assert relative_error < 1e-4


def test_fit_can_share_one_vocabulary_factor_between_embeddings() -> None:
    torch.manual_seed(15)
    shared = torch.randn(64, 4)
    input_right = torch.randn(8, 4)
    output_right = torch.randn(8, 4)
    w0 = {
        "embed_out.weight": torch.randn(64, 8),
        "gpt_neox.embed_in.weight": torch.randn(64, 8),
    }
    wt = {
        "embed_out.weight": w0["embed_out.weight"] + shared @ output_right.transpose(0, 1),
        "gpt_neox.embed_in.weight": (
            w0["gpt_neox.embed_in.weight"] + shared @ input_right.transpose(0, 1)
        ),
    }
    graph = graph_from_state(w0, family="toy", config={})

    program, payloads = fit_low_rank_program(
        w0,
        wt,
        graph,
        config=FitConfig(
            budget_fraction=0.50,
            max_rank=4,
            allocation_strategy="rank_balanced",
            shared_vocabulary_factors=True,
            svd_method="exact",
        ),
    )
    candidate = execute_program(w0, program, payloads)
    left_keys = {
        component.payload["left"]
        for tensor in program.tensors
        for component in tensor.components
        if component.primitive == "LOW_RANK"
    }

    assert left_keys == {"shared.vocabulary.low_rank.left"}
    assert torch.allclose(candidate["embed_out.weight"], wt["embed_out.weight"], atol=0.02)
    assert torch.allclose(
        candidate["gpt_neox.embed_in.weight"],
        wt["gpt_neox.embed_in.weight"],
        atol=0.02,
    )


def test_internal_first_allocation_prioritizes_transformer_matrices() -> None:
    torch.manual_seed(16)
    w0 = {
        "embed_out.weight": torch.randn(64, 8),
        "gpt_neox.embed_in.weight": torch.randn(64, 8),
        "layers.0.weight": torch.randn(16, 16),
        "layers.1.weight": torch.randn(16, 16),
    }
    wt = {name: value + torch.randn_like(value) for name, value in w0.items()}
    graph = graph_from_state(w0, family="toy", config={})

    program, _ = fit_low_rank_program(
        w0,
        wt,
        graph,
        config=FitConfig(
            budget_fraction=0.20,
            max_rank=4,
            minimum_matrix_rank=1,
            allocation_strategy="internal_first",
            shared_vocabulary_factors=True,
            svd_method="exact",
        ),
    )
    ranks = {
        tensor.name: int(component.arguments["rank"])
        for tensor in program.tensors
        for component in tensor.components
        if component.primitive == "LOW_RANK"
    }

    assert ranks["layers.0.weight"] > ranks["embed_out.weight"]
    assert ranks["layers.1.weight"] > ranks["gpt_neox.embed_in.weight"]


def test_all_floating_program_coefficients_are_trainable() -> None:
    program = ModelGenomeProgram(
        architecture_id="a",
        base_state_id="b",
        tensors=(
            TensorProgram(
                name="vector",
                shape=(2,),
                components=(
                    Component("BASE_COPY"),
                    Component(
                        "QUANTIZED_VECTOR",
                        payload={"values": "vector.values", "scale": "vector.scale"},
                    ),
                ),
            ),
        ),
    )
    trainable = TrainableProgram(
        program,
        {
            "vector.values": torch.tensor([1, -1], dtype=torch.int8),
            "vector.scale": torch.tensor([0.5]),
        },
    )

    assert set(trainable.parameters_by_key) == {"vector__scale"}
    assert set(trainable.constants) == {"vector.values"}


def test_trainable_program_restores_compact_storage_dtypes() -> None:
    program = ModelGenomeProgram(
        architecture_id="a",
        base_state_id="b",
        tensors=(
            TensorProgram(
                name="matrix",
                shape=(2, 2),
                components=(
                    Component("BASE_COPY"),
                    Component(
                        "LOW_RANK",
                        payload={"left": "matrix.left", "right": "matrix.right"},
                        arguments={"rank": 1},
                    ),
                ),
            ),
        ),
    )
    trainable = TrainableProgram(
        program,
        {
            "matrix.left": torch.ones(2, 1, dtype=torch.float16),
            "matrix.right": torch.ones(2, 1, dtype=torch.float16),
        },
    )

    assert trainable.payloads()["matrix.left"].dtype == torch.float32
    assert trainable.export_payloads()["matrix.left"].dtype == torch.float16


def test_teacher_kl_is_averaged_per_token() -> None:
    student = torch.tensor([[[2.0, -1.0]]])
    teacher = torch.tensor([[[1.0, 0.0]]])
    expected = _token_mean_kl(student, teacher)

    repeated = _token_mean_kl(student.repeat(1, 8, 1), teacher.repeat(1, 8, 1))

    assert repeated == pytest.approx(float(expected))


def test_relative_anchor_loss_uses_one_global_scale() -> None:
    anchors = {
        "large": torch.tensor([2.0, 2.0]),
        "zero": torch.tensor([0.0]),
    }
    parameters = {
        "large": torch.tensor([3.0, 1.0]),
        "zero": torch.tensor([1.0]),
    }

    loss = _relative_anchor_loss(parameters, anchors)

    assert loss == pytest.approx(3.0 / 8.0)
    assert _relative_anchor_loss(anchors, anchors) == 0


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
