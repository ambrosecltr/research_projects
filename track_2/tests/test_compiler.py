from __future__ import annotations

import torch

from genome.architecture import graph_from_state
from genome.compiler.data import build_compiler_example, recipe_vector
from genome.compiler.model import CompilerConfig, GenomeCompiler, compiler_loss
from genome.compiler.train import labels_from_program
from genome.fingerprint import corpus_fingerprint
from genome.state import direct_fp16_delta_bytes, state_id


def example_and_targets():
    torch.manual_seed(3)
    w0 = {
        "layers.0.weight": torch.randn(48, 32),
        "layers.0.bias": torch.randn(48),
        "layers.1.weight": torch.randn(32, 48),
        "final_norm.weight": torch.ones(32),
    }
    graph = graph_from_state(w0, family="toy", config={"hidden": 32})
    fingerprint = corpus_fingerprint([[1, 2, 3], [4, 5, 6]])
    config = CompilerConfig(
        global_feature_dim=32,
        tensor_feature_dim=16,
        coordinate_feature_dim=4,
        d_model=32,
        n_heads=4,
        transformer_layers=1,
        message_layers=1,
        max_rank=4,
        target_fraction=0.2,
        manifest_reserve_bytes=0,
    )
    example = build_compiler_example(
        graph,
        w0,
        fingerprint,
        {"optimizer": {"lr": 1e-3}, "tokens": 1000},
        global_feature_dim=config.global_feature_dim,
        tensor_feature_dim=config.tensor_feature_dim,
        base_state_id=state_id(w0),
    )
    targets = {name: torch.randn_like(value) * 0.01 for name, value in w0.items()}
    return config, example, w0, targets


def test_hierarchical_compiler_forward_backward_and_program() -> None:
    config, example, w0, targets = example_and_targets()
    compiler = GenomeCompiler(config)
    prediction = compiler(example)
    assert prediction.contexts.shape == (len(example.tensors), config.d_model)
    primitives = torch.tensor([1, 2, 1, 2])
    ranks = torch.tensor([2, 0, 2, 0])
    loss, metrics = compiler_loss(
        compiler,
        example,
        target_primitives=primitives,
        target_ranks=ranks,
        target_deltas=targets,
        w0_state=w0,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert metrics["expected_bytes"] >= 0
    assert 0.0 <= metrics["primitive_accuracy"] <= 1.0
    assert 0.0 <= metrics["rank_accuracy"] <= 1.0
    program, payloads = compiler.generate_program(
        example,
        direct_fp16_delta_bytes=direct_fp16_delta_bytes(w0),
    )
    assert len(program.tensors) == len(w0)
    assert all(
        value.numel() < sum(item.numel() for item in w0.values()) for value in payloads.values()
    )
    assert all(
        component.primitive != "SPARSE_PATCH"
        for tensor in program.tensors
        for component in tensor.components
    )


def test_compiler_emits_formula_v2_matrix_and_vector_components() -> None:
    config, example, w0, _ = example_and_targets()
    compiler = GenomeCompiler(config)
    with torch.no_grad():
        compiler.primitive_head.weight.zero_()
        compiler.primitive_head.bias.copy_(torch.tensor([0.0, 2.0, 1.0]))
        compiler.rank_head.weight.zero_()
        compiler.rank_head.bias.copy_(torch.tensor([0.0, 0.0, 2.0, 0.0, 0.0]))
    program, payloads = compiler.generate_program(
        example,
        direct_fp16_delta_bytes=direct_fp16_delta_bytes(w0),
    )
    by_name = {tensor.name: tensor for tensor in program.tensors}
    matrix_primitives = {component.primitive for component in by_name["layers.0.weight"].components}
    assert {"BASE_COPY", "HADAMARD_SCALE", "LOW_RANK"} <= matrix_primitives
    matrix_scale = next(
        component
        for component in by_name["layers.0.weight"].components
        if component.primitive == "HADAMARD_SCALE"
    )
    assert payloads[matrix_scale.payload["row"]].dtype == torch.float16

    with torch.no_grad():
        compiler.primitive_head.bias.copy_(torch.tensor([0.0, 1.0, 2.0]))
    program, payloads = compiler.generate_program(
        example,
        direct_fp16_delta_bytes=direct_fp16_delta_bytes(w0),
    )
    by_name = {tensor.name: tensor for tensor in program.tensors}
    vector_primitives = {component.primitive for component in by_name["layers.0.bias"].components}
    assert "DIRECT_VECTOR" in vector_primitives
    vector = next(
        component
        for component in by_name["layers.0.bias"].components
        if component.primitive == "DIRECT_VECTOR"
    )
    assert payloads[vector.payload["values"]].dtype == torch.float16
    vector_index = next(
        index for index, tensor in enumerate(program.tensors) if tensor.name == "layers.0.bias"
    )
    assert labels_from_program(program).primitives[vector_index] == 2


def test_output_packet_count_is_bounded_by_program_budget() -> None:
    config, example, w0, _ = example_and_targets()
    compiler = GenomeCompiler(config)
    _, payloads = compiler.generate_program(
        example, direct_fp16_delta_bytes=direct_fp16_delta_bytes(w0)
    )
    payload_bytes = sum(item.numel() * item.element_size() for item in payloads.values())
    assert payload_bytes <= int(direct_fp16_delta_bytes(w0) * config.target_fraction)


def test_compiler_reuses_one_vocabulary_left_factor() -> None:
    torch.manual_seed(23)
    w0 = {
        "embed_out.weight": torch.randn(20, 4),
        "gpt_neox.embed_in.weight": torch.randn(20, 4),
    }
    graph = graph_from_state(w0, family="toy", config={"hidden": 4})
    fingerprint = corpus_fingerprint([[1, 2, 3]])
    config = CompilerConfig(
        global_feature_dim=16,
        tensor_feature_dim=12,
        coordinate_feature_dim=4,
        d_model=16,
        n_heads=4,
        transformer_layers=1,
        message_layers=1,
        max_rank=2,
        target_fraction=0.8,
        manifest_reserve_bytes=0,
        shared_vocabulary_factors=True,
    )
    example = build_compiler_example(
        graph,
        w0,
        fingerprint,
        {"tokens": 100},
        global_feature_dim=config.global_feature_dim,
        tensor_feature_dim=config.tensor_feature_dim,
        base_state_id=state_id(w0),
    )
    compiler = GenomeCompiler(config)
    with torch.no_grad():
        compiler.primitive_head.weight.zero_()
        compiler.primitive_head.bias.copy_(torch.tensor([0.0, 2.0, 1.0]))
        compiler.rank_head.weight.zero_()
        compiler.rank_head.bias.copy_(torch.tensor([0.0, 0.0, 2.0]))
    program, payloads = compiler.generate_program(
        example,
        direct_fp16_delta_bytes=direct_fp16_delta_bytes(w0),
    )
    left_payloads = {
        component.payload["left"]
        for tensor in program.tensors
        for component in tensor.components
        if component.primitive == "LOW_RANK"
    }
    assert left_payloads == {"shared.vocabulary.low_rank.left"}
    payload_bytes = sum(item.numel() * item.element_size() for item in payloads.values())
    assert payload_bytes <= int(direct_fp16_delta_bytes(w0) * config.target_fraction)
    loss, _ = compiler_loss(
        compiler,
        example,
        target_primitives=torch.tensor([1, 1]),
        target_ranks=torch.tensor([2, 2]),
        target_deltas={name: torch.randn_like(value) * 0.01 for name, value in w0.items()},
        w0_state=w0,
    )
    loss.backward()
    assert compiler.left_head.network[0].weight.grad is not None


def test_recipe_vector_excludes_provenance_but_keeps_data_order_seed() -> None:
    recipe = {
        "dataset": {
            "dataset_id": "pile-standard",
            "repository": "EleutherAI/pile",
            "revision": "a" * 40,
            "order_id": "seed9-at-a",
            "data_order_seed": 9,
        },
        "stage": {"steps": 143000, "tokens": 299892736000},
    }
    changed_provenance = {
        **recipe,
        "dataset": {
            **recipe["dataset"],
            "repository": "different/repository",
            "revision": "b" * 40,
            "order_id": "seed9-at-b",
        },
    }
    changed_order_seed = {
        **recipe,
        "dataset": {**recipe["dataset"], "data_order_seed": 8},
    }
    base = recipe_vector(recipe, 128)
    assert torch.equal(base, recipe_vector(changed_provenance, 128))
    assert not torch.equal(base, recipe_vector(changed_order_seed, 128))
