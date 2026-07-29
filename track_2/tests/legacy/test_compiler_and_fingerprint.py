from __future__ import annotations

import pytest
import torch

from genome.fingerprint import count_sketch
from genome.neural.compiler import GenomeCodeLayout, GenomeCompiler

pytestmark = pytest.mark.legacy


def test_count_sketch_is_deterministic_and_seeded():
    value = torch.arange(100, dtype=torch.float32)
    a = count_sketch(value, 16, seed=7)
    b = count_sketch(value, 16, seed=7)
    c = count_sketch(value, 16, seed=8)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_compiler_output_layout():
    layout = GenomeCodeLayout(
        global_code_dim=4,
        n_layers=3,
        layer_code_dim=2,
        n_tensors=5,
        tensor_code_dim=3,
    )
    model = GenomeCompiler(
        architecture_dim=6,
        dataset_fingerprint_dim=8,
        trajectory_fingerprint_dim=10,
        layout=layout,
        hidden_dim=32,
        depth=2,
    )
    distribution = model(torch.randn(2, 6), torch.randn(2, 8), torch.randn(2, 10))
    codes = distribution.mode()
    assert codes["global_code"].shape == (2, 4)
    assert codes["layer_codes"].shape == (2, 3, 2)
    assert codes["tensor_codes"].shape == (2, 5, 3)
