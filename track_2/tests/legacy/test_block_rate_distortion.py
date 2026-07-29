from __future__ import annotations

import pytest
import torch

from genome.io import read_json
from genome.neural.block_rate_distortion import (
    analyze_tensor_svd_rate_distortion,
    summarize_centered_spectrum,
)
from genome.types import TensorSpec

pytestmark = pytest.mark.legacy


class _Life:
    split = "training"

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id

    @staticmethod
    def load_base() -> dict[str, torch.Tensor]:
        return {"matrix": torch.zeros(2, 2), "vector": torch.zeros(1)}

    @staticmethod
    def load_target() -> dict[str, torch.Tensor]:
        return {
            "matrix": torch.tensor([[3.0, 0.0], [0.0, 1.0]]),
            "vector": torch.tensor([2.0]),
        }


def test_centered_spectrum_reports_exact_rate_points() -> None:
    result = summarize_centered_spectrum(
        torch.tensor([0.0, 1.0, 3.0]),
        sample_count=10,
        valid_value_count=30,
        blocks_per_life=5,
        widths=(0, 1, 2, 3),
    )

    assert result["effective_rank"] > 1.0
    points = result["rate_points"]
    assert points[0]["relative_residual_energy"] == 1.0
    assert points[1]["explained_centered_energy"] == 0.75
    assert points[2]["explained_centered_energy"] == 1.0
    assert points[3]["normalized_mse_per_valid_value"] == 0.0
    assert points[1]["fp16_code_bytes_per_life"] == 10


def test_centered_spectrum_rejects_width_larger_than_block() -> None:
    try:
        summarize_centered_spectrum(
            torch.ones(4),
            sample_count=2,
            valid_value_count=8,
            blocks_per_life=1,
            widths=(5,),
        )
    except ValueError as error:
        assert str(error) == "block-code width cannot exceed the flattened block size"
    else:
        raise AssertionError("invalid width was accepted")


def test_tensor_svd_rate_distortion_is_exact(tmp_path) -> None:
    specs = [
        TensorSpec(
            canonical_index=0,
            name="matrix",
            role="weight",
            layer_index=0,
            shape=(2, 2),
            dtype="float32",
            numel=4,
            nbytes=16,
        ),
        TensorSpec(
            canonical_index=1,
            name="vector",
            role="bias",
            layer_index=0,
            shape=(1,),
            dtype="float32",
            numel=1,
            nbytes=4,
        ),
    ]
    output = tmp_path / "tensor-rate.json"
    result = analyze_tensor_svd_rate_distortion(
        [_Life("seed0"), _Life("seed1")],
        tensor_specs=specs,
        tied_groups=[],
        ranks=(0, 1, 2),
        device="cpu",
        output_path=output,
    )

    points = result["aggregate_rate_points"]
    assert points[1]["explained_matrix_delta_energy"] == pytest.approx(0.9)
    assert points[1]["fp16_factor_bytes_per_life"] == 10
    assert points[2]["explained_matrix_delta_energy"] == 1.0
    assert result["direct_fp16_non_matrix_bytes_per_life"] == 2
    assert read_json(output)["content_sha256"] == result["content_sha256"]
