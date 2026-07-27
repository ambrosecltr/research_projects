"""Apply only an accepted weights forecast; optimizer state is deliberately absent."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from poetry50m.trajectory.verification import CandidateVerification


def apply_accepted_candidate(module: nn.Module, verification: CandidateVerification) -> None:
    """Atomically load a verified state dict into ``module``.

    Forecasts contain model weights/buffers only. Optimizer moments are neither
    loaded nor extrapolated; the caller may retain its normal optimizer state or
    explicitly reset it as a separately recorded training decision.
    """

    if not verification.decision.accepted:
        raise ValueError("refusing to apply a candidate that failed acceptance gates")
    if verification.decision.candidate_state_hash != verification.prepared.state_hash:
        raise ValueError("candidate was not verified in the exact state proposed for application")
    current = module.state_dict()
    candidate = verification.prepared.state_dict
    if tuple(current) != tuple(candidate):
        raise ValueError("forecast state_dict names or order do not match the live module")
    prepared: dict[str, Tensor] = {}
    for name, current_tensor in current.items():
        proposed = candidate[name]
        if current_tensor.shape != proposed.shape or current_tensor.dtype != proposed.dtype:
            raise ValueError(f"forecast tensor {name} does not match the live model coordinate")
        if proposed.is_floating_point() and not bool(torch.isfinite(proposed).all().item()):
            raise ValueError(f"forecast tensor {name} contains non-finite values")
        prepared[name] = proposed.detach().to(
            device=current_tensor.device, dtype=current_tensor.dtype
        )
    module.load_state_dict(prepared, strict=True)
