from __future__ import annotations

import torch

from genome.architecture import graph_from_state
from genome.compiler.data import build_compiler_example
from genome.compiler.model import CompilerConfig, GenomeCompiler, compiler_loss
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
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert metrics["expected_bytes"] >= 0
    program, payloads = compiler.generate_program(
        example,
        direct_fp16_delta_bytes=direct_fp16_delta_bytes(w0),
    )
    assert len(program.tensors) == len(w0)
    assert all(value.numel() < sum(item.numel() for item in w0.values()) for value in payloads.values())
    assert all(component.primitive != "SPARSE_PATCH" for tensor in program.tensors for component in tensor.components)


def test_output_packet_count_is_bounded_by_program_budget() -> None:
    config, example, w0, _ = example_and_targets()
    compiler = GenomeCompiler(config)
    _, payloads = compiler.generate_program(example, direct_fp16_delta_bytes=direct_fp16_delta_bytes(w0))
    payload_bytes = sum(item.numel() * item.element_size() for item in payloads.values())
    assert payload_bytes <= int(direct_fp16_delta_bytes(w0) * config.target_fraction)
