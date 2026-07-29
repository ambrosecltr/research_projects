from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ProgramLengthEstimate:
    model: str
    parameter_count: int
    coefficient_chunk_dim: int
    median_budget_fraction: float
    upper_budget_fraction: float
    median_tokens: int
    upper_tokens: int
    max_program_tokens: int

    @property
    def median_exceeds_limit(self) -> bool:
        return self.median_tokens > self.max_program_tokens

    @property
    def upper_exceeds_limit(self) -> bool:
        return self.upper_tokens > self.max_program_tokens

    def to_dict(self) -> dict[str, int | float | str | bool]:
        return {
            **asdict(self),
            "median_exceeds_limit": self.median_exceeds_limit,
            "upper_exceeds_limit": self.upper_exceeds_limit,
        }


def estimate_program_tokens(
    *,
    parameter_count: int,
    tensor_count: int,
    target_fraction_of_fp16_delta: float,
    coefficient_chunk_dim: int = 16,
) -> int:
    """Estimate current flat-token length from an fp16 target byte budget.

    One numeric token carries ``coefficient_chunk_dim`` fp16 target coefficients. The structural
    allowance is four tokens per tensor plus BOS/EOS. This is an analytic capacity estimate, not a
    substitute for tokenizing a fitted target.
    """

    if (
        isinstance(parameter_count, bool)
        or not isinstance(parameter_count, int)
        or parameter_count < 1
    ):
        raise ValueError("parameter_count must be a positive integer")
    if isinstance(tensor_count, bool) or not isinstance(tensor_count, int) or tensor_count < 1:
        raise ValueError("tensor_count must be a positive integer")
    if not 0.0 < target_fraction_of_fp16_delta <= 1.0:
        raise ValueError("target_fraction_of_fp16_delta must lie in (0, 1]")
    if (
        isinstance(coefficient_chunk_dim, bool)
        or not isinstance(coefficient_chunk_dim, int)
        or coefficient_chunk_dim < 1
    ):
        raise ValueError("coefficient_chunk_dim must be a positive integer")
    payload_values = math.floor(parameter_count * target_fraction_of_fp16_delta)
    payload_tokens = math.ceil(payload_values / coefficient_chunk_dim)
    return payload_tokens + tensor_count * 4 + 2


def pythia_program_length_estimates(
    *,
    coefficient_chunk_dim: int = 16,
    max_program_tokens: int = 4096,
) -> tuple[ProgramLengthEstimate, ...]:
    """Return flat-token estimates for the official Pythia 14M and 31M shapes.

    The exact parameter counts follow the untied GPT-NeoX configurations: six layers, vocabulary
    50,304, and hidden widths 128 or 256. Each model has 76 trainable tensor records.
    """

    models = (
        ("pythia-14m", 14_067_712),
        ("pythia-31m", 30_494_720),
    )
    median_fraction = 0.05
    upper_fraction = 0.10
    return tuple(
        ProgramLengthEstimate(
            model=model,
            parameter_count=parameter_count,
            coefficient_chunk_dim=coefficient_chunk_dim,
            median_budget_fraction=median_fraction,
            upper_budget_fraction=upper_fraction,
            median_tokens=estimate_program_tokens(
                parameter_count=parameter_count,
                tensor_count=76,
                target_fraction_of_fp16_delta=median_fraction,
                coefficient_chunk_dim=coefficient_chunk_dim,
            ),
            upper_tokens=estimate_program_tokens(
                parameter_count=parameter_count,
                tensor_count=76,
                target_fraction_of_fp16_delta=upper_fraction,
                coefficient_chunk_dim=coefficient_chunk_dim,
            ),
            max_program_tokens=max_program_tokens,
        )
        for model, parameter_count in models
    )
