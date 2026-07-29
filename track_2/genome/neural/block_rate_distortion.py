from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from ..hashing import sha256_json
from ..io import write_json
from ..polypythia.lives import CanonicalModelLife
from ..types import TensorSpec
from .block_decoder import BlockDecoderConfig
from .multilife_decoder import MultiLifeBlockSampler


def _validate_widths(widths: Sequence[int], block_values: int) -> tuple[int, ...]:
    result = tuple(sorted(set(widths)))
    if not result:
        raise ValueError("at least one block-code width is required")
    if any(
        isinstance(width, bool) or not isinstance(width, int) or width < 0
        for width in result
    ):
        raise ValueError("block-code widths must be non-negative integers")
    if result[-1] > block_values:
        raise ValueError("block-code width cannot exceed the flattened block size")
    return result


def summarize_centered_spectrum(
    eigenvalues: torch.Tensor,
    *,
    sample_count: int,
    valid_value_count: int,
    blocks_per_life: int,
    widths: Sequence[int],
) -> dict[str, Any]:
    if eigenvalues.ndim != 1 or eigenvalues.numel() < 1:
        raise ValueError("eigenvalues must be a non-empty vector")
    if sample_count < 2 or valid_value_count < sample_count or blocks_per_life < 1:
        raise ValueError("invalid block-spectrum counts")
    checked_widths = _validate_widths(widths, eigenvalues.numel())
    descending = eigenvalues.detach().to(torch.float64).clamp_min(0).flip(0)
    total = float(descending.sum().item())
    mean_valid_values = valid_value_count / sample_count
    if total <= 0:
        effective_rank = 0.0
    else:
        probabilities = descending / total
        nonzero = probabilities[probabilities > 0]
        effective_rank = float(torch.exp(-(nonzero * nonzero.log()).sum()).item())
    rate_points = []
    for width in checked_widths:
        retained = float(descending[:width].sum().item())
        residual = max(total - retained, 0.0)
        rate_points.append(
            {
                "code_width": width,
                "explained_centered_energy": 1.0 if total == 0 else retained / total,
                "relative_residual_energy": 0.0 if total == 0 else residual / total,
                "normalized_mse_per_valid_value": residual / mean_valid_values,
                "fp16_code_bytes_per_life": blocks_per_life * width * 2,
                "fp32_code_bytes_per_life": blocks_per_life * width * 4,
            }
        )
    return {
        "sample_count": sample_count,
        "valid_value_count": valid_value_count,
        "mean_valid_values_per_block": mean_valid_values,
        "blocks_per_life": blocks_per_life,
        "centered_energy": total,
        "effective_rank": effective_rank,
        "rate_points": rate_points,
    }


def analyze_block_rate_distortion(
    lives: Sequence[CanonicalModelLife],
    *,
    tensor_specs: Sequence[TensorSpec],
    tied_groups: Sequence[Sequence[str]],
    decoder_config: BlockDecoderConfig,
    widths: Sequence[int],
    batch_size: int,
    device: str | torch.device,
    output_path: str | Path,
) -> dict[str, Any]:
    if not lives or any(life.split != "training" for life in lives):
        raise ValueError("block analysis requires training-split model lives")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    block_values = decoder_config.block_rows * decoder_config.block_cols
    checked_widths = _validate_widths(widths, block_values)
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(destination)

    base_states = [life.load_base() for life in lives]
    target_states = [life.load_target() for life in lives]
    sampler = MultiLifeBlockSampler(
        base_states=base_states,
        target_states=target_states,
        tensor_specs=tensor_specs,
        tied_groups=tied_groups,
        decoder_config=decoder_config,
    )
    active_device = torch.device(device)
    role_results: dict[str, Any] = {}
    aggregate_valid_values = 0
    aggregate_samples = 0

    for role in sampler.roles:
        reference_indices = sampler.references_by_role[role]
        value_sum = torch.zeros(block_values, dtype=torch.float64, device=active_device)
        value_gram = torch.zeros(
            block_values,
            block_values,
            dtype=torch.float64,
            device=active_device,
        )
        valid_value_count = 0
        sample_count = 0
        for life_index in range(sampler.life_count):
            for start in range(0, len(reference_indices), batch_size):
                selected = reference_indices[start : start + batch_size]
                batch = sampler.make_indexed_batch(
                    life_indices=torch.full((len(selected),), life_index, dtype=torch.long),
                    reference_indices=torch.tensor(selected, dtype=torch.long),
                    device=active_device,
                )
                values = batch.targets.flatten(1).to(torch.float64)
                value_sum += values.sum(dim=0)
                value_gram += values.transpose(0, 1) @ values
                valid_value_count += int(batch.valid_masks.sum().item())
                sample_count += values.shape[0]

        mean = value_sum / sample_count
        covariance = value_gram / sample_count - torch.outer(mean, mean)
        covariance = (covariance + covariance.transpose(0, 1)) * 0.5
        eigenvalues = torch.linalg.eigvalsh(covariance).cpu()
        result = summarize_centered_spectrum(
            eigenvalues,
            sample_count=sample_count,
            valid_value_count=valid_value_count,
            blocks_per_life=len(reference_indices),
            widths=checked_widths,
        )
        result["tensor_role"] = role
        role_results[role] = result
        aggregate_valid_values += valid_value_count
        aggregate_samples += sample_count

    aggregate_points = []
    for width in checked_widths:
        centered_energy = 0.0
        retained_energy = 0.0
        fp16_bytes = 0
        fp32_bytes = 0
        for result in role_results.values():
            point = next(
                item for item in result["rate_points"] if item["code_width"] == width
            )
            role_energy = float(result["centered_energy"])
            centered_energy += role_energy * int(result["sample_count"])
            retained_energy += (
                role_energy
                * int(result["sample_count"])
                * float(point["explained_centered_energy"])
            )
            fp16_bytes += int(point["fp16_code_bytes_per_life"])
            fp32_bytes += int(point["fp32_code_bytes_per_life"])
        residual_energy = max(centered_energy - retained_energy, 0.0)
        aggregate_points.append(
            {
                "code_width_per_block": width,
                "explained_centered_energy": (
                    1.0 if centered_energy == 0 else retained_energy / centered_energy
                ),
                "relative_residual_energy": (
                    0.0 if centered_energy == 0 else residual_energy / centered_energy
                ),
                "normalized_mse_per_valid_value": (
                    residual_energy / aggregate_valid_values
                ),
                "fp16_code_bytes_per_life": fp16_bytes,
                "fp32_code_bytes_per_life": fp32_bytes,
            }
        )

    result = {
        "format": "GENOME_BLOCK_RATE_DISTORTION",
        "version": "0.1.0",
        "method": "exact_role_conditioned_centered_pca",
        "training_run_ids": [life.run_id for life in lives],
        "hidden_endpoints_seen": False,
        "decoder_block_shape": [
            decoder_config.block_rows,
            decoder_config.block_cols,
        ],
        "widths": list(checked_widths),
        "life_count": sampler.life_count,
        "untied_blocks_per_life": len(sampler.references),
        "sample_count": aggregate_samples,
        "valid_value_count": aggregate_valid_values,
        "roles": role_results,
        "aggregate_rate_points": aggregate_points,
    }
    result["content_sha256"] = sha256_json(result)
    write_json(destination, result, canonical=True)
    return result
