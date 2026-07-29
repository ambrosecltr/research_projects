from __future__ import annotations

from genome.program_scalability import pythia_program_length_estimates


def test_pythia_flat_program_estimator_cannot_hide_context_overflow() -> None:
    estimates = pythia_program_length_estimates()
    assert [estimate.model for estimate in estimates] == ["pythia-14m", "pythia-31m"]
    assert [estimate.median_tokens for estimate in estimates] == [44_268, 95_602]
    assert [estimate.upper_tokens for estimate in estimates] == [88_230, 190_898]
    assert all(estimate.median_exceeds_limit for estimate in estimates)
    assert all(estimate.upper_exceeds_limit for estimate in estimates)
