from __future__ import annotations

import torch

from genome.neural.block_rate_distortion import summarize_centered_spectrum


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
